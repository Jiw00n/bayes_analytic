/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#include <tvm/arith/int_set.h>
#include <tvm/arith/var_context.h>
#include <tvm/auto_scheduler/compute_dag.h>
#include <tvm/auto_scheduler/loop_state.h>
#include <tvm/auto_scheduler/search_task.h>
#include <tvm/auto_scheduler/transform_step.h>
#include <tvm/ir/module.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/registry.h>
#include <tvm/te/operation.h>
#include <tvm/te/schedule_pass.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/buffer.h>
#include <tvm/tir/function.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <string>
#include <limits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../runtime/thread_storage_scope.h"
#include "../tir/transforms/ir_utils.h"

namespace tvm {
namespace auto_scheduler {
namespace {

using tir::For;
using tir::ForKind;
using tir::IterVar;
using tir::PrimFunc;
using tir::Stmt;
using tir::StmtExprMutator;
using tir::StmtExprVisitor;
using tir::Substitute;

struct SymbolicSplitContext {
  arith::VarContext vcontext;
  std::unordered_map<int, Array<PrimExpr>> split_lengths;
  Map<tir::Var, PrimExpr> rename_map;
};

void GetBinds(const Array<ObjectRef>& args, bool compact,
              const std::unordered_map<te::Tensor, tir::Buffer>& binds,
              Map<te::Tensor, tir::Buffer>* out_binds, Array<ObjectRef>* out_arg_list) {
  *out_binds = binds;

  for (const ObjectRef& x : args) {
    if (auto tensor_node = x.as<te::TensorNode>()) {
      te::Tensor x_ref = GetRef<te::Tensor>(tensor_node);
      if (out_binds->find(x_ref) == out_binds->end()) {
        tir::Buffer buf = tir::BufferWithOffsetAlignment(x_ref->shape, x_ref->dtype,
                                                         x_ref->op->name, -1, 0, compact);
        out_binds->Set(x_ref, buf);
        out_arg_list->push_back(buf);
      } else {
        out_arg_list->push_back((*out_binds)[x_ref]);
      }
    } else {
      out_arg_list->push_back(x);
    }
  }
}

Array<te::Operation> GetOutputOps(const ComputeDAG& dag) {
  Array<te::Operation> out_ops;
  for (const auto& op : dag->ops) {
    if (dag->access_analyzer.IsOutput(op)) {
      out_ops.push_back(op);
    }
  }
  return out_ops;
}

std::pair<te::Schedule, Array<te::Tensor>> BuildInitialSchedule(const ComputeDAG& dag,
                                                                Array<te::Stage>* stages,
                                                                StageToAxesMap* stage_to_axes) {
  te::Schedule schedule = te::create_schedule(GetOutputOps(dag));
  for (const auto& op : dag->ops) {
    const te::Stage& stage = schedule[op];
    stages->push_back(stage);
    UpdateStageToAxesMap(stage, stage_to_axes);
  }
  return {schedule, dag->tensors};
}

Array<PrimExpr> ApplySplitToScheduleSymbolic(Array<te::Stage>* stages, StageToAxesMap* stage_to_axes,
                                             int stage_id, int iter_id,
                                             const Array<PrimExpr>& lengths,
                                             bool inner_to_outer) {
  ICHECK_GE(stage_id, 0);
  ICHECK_LT(stage_id, static_cast<int>(stages->size()))
      << "Invalid stage_id=" << stage_id << " for " << stages->size() << " stages";
  auto stage = (*stages)[stage_id];
  auto it = stage_to_axes->find(stage);
  ICHECK(it != stage_to_axes->end()) << "Missing stage_to_axes entry for stage_id=" << stage_id
                                     << " op=" << stage->op;
  Array<IterVar> axes = (*it).second;
  ICHECK_GE(iter_id, 0);
  ICHECK_LT(iter_id, static_cast<int>(axes.size()))
      << "Invalid iter_id=" << iter_id << " for stage_id=" << stage_id
      << " with " << axes.size() << " axes";
  Array<IterVar> outs;
  if (inner_to_outer) {
    IterVar outer = axes[iter_id], inner;
    for (int i = static_cast<int>(lengths.size()) - 1; i >= 0; --i) {
      IterVar to_split = outer;
      stage.split(to_split, lengths[i], &outer, &inner);
      outs.push_back(inner);
    }
    outs.push_back(outer);
  } else {
    IterVar outer, inner = axes[iter_id];
    for (size_t i = 0; i < lengths.size(); ++i) {
      IterVar to_split = inner;
      stage.split_by_nparts(to_split, lengths[i], &outer, &inner);
      outs.push_back(outer);
    }
    outs.push_back(inner);
  }

  Array<IterVar> new_axes;
  new_axes.insert(new_axes.end(), axes.begin(), axes.begin() + iter_id);
  if (inner_to_outer) {
    for (auto x = outs.rbegin(); x != outs.rend(); ++x) {
      new_axes.push_back(*x);
    }
  } else {
    for (const auto& x : outs) {
      new_axes.push_back(x);
    }
  }
  new_axes.insert(new_axes.end(), axes.begin() + iter_id + 1, axes.end());

  stage_to_axes->Set(stage, std::move(new_axes));
  stages->Set(stage_id, std::move(stage));

  return Array<PrimExpr>();
}

Array<PrimExpr> MakeSplitLengthsForStep(int step_id, const SplitStepNode* ps,
                                        SymbolicSplitContext* ctx) {
  ICHECK(ps->extent.defined()) << "SplitStep without extent is unsupported in symbolic lowering";

  Array<tir::SizeVar> internal =
      ctx->vcontext.GetSplitVars(ps->extent.value(), ps->lengths.size(), true);
  Array<PrimExpr> result;
  for (size_t i = 0; i < internal.size(); ++i) {
    tir::SizeVar renamed("sp_" + std::to_string(step_id) + "_" + std::to_string(i));
    ctx->rename_map.Set(internal[i], renamed);
    result.push_back(internal[i]);
  }
  ctx->split_lengths[step_id] = result;
  return result;
}

Array<PrimExpr> ExtractFollowSplitLengths(const FollowSplitStepNode* ps,
                                         const SymbolicSplitContext& ctx) {
  auto it = ctx.split_lengths.find(ps->src_step_id);
  ICHECK(it != ctx.split_lengths.end()) << "Missing source split step " << ps->src_step_id;
  const Array<PrimExpr>& src_lengths = it->second;
  ICHECK_LE(ps->n_split, src_lengths.size() + 1);

  Array<PrimExpr> lengths;
  int j = 0;
  for (; j < ps->n_split - 1; ++j) {
    lengths.push_back(src_lengths[j]);
  }

  PrimExpr last_factor = 1;
  for (; j < static_cast<int>(src_lengths.size()); ++j) {
    last_factor *= src_lengths[j];
  }
  lengths.push_back(last_factor);
  return lengths;
}

PrimExpr ExtractFollowFusedSplitLength(const FollowFusedSplitStepNode* ps,
                                       const SymbolicSplitContext& ctx) {
  PrimExpr ret = 1;
  for (const auto& src_step_id : ps->src_step_ids) {
    auto it = ctx.split_lengths.find(src_step_id.IntValue());
    ICHECK(it != ctx.split_lengths.end()) << "Missing source split step " << src_step_id;
    ICHECK_LT(ps->level, static_cast<int>(it->second.size()));
    ret *= it->second[ps->level];
  }
  return ret;
}

void StepApplyToScheduleSymbolic(const Step& step, int step_id, Array<te::Stage>* stages,
                                 StageToAxesMap* stage_to_axes, te::Schedule* schedule,
                                 SymbolicSplitContext* split_ctx,
                                 const Array<Step>& transform_steps) {
  if (auto ps = step.as<SplitStepNode>()) {
    Array<PrimExpr> lengths = MakeSplitLengthsForStep(step_id, ps, split_ctx);
    ApplySplitToScheduleSymbolic(stages, stage_to_axes, ps->stage_id, ps->iter_id, lengths,
                                 ps->inner_to_outer);
  } else if (auto ps = step.as<FollowSplitStepNode>()) {
    Array<PrimExpr> lengths = ExtractFollowSplitLengths(ps, *split_ctx);
    ApplySplitToScheduleSymbolic(stages, stage_to_axes, ps->stage_id, ps->iter_id, lengths, true);
  } else if (auto ps = step.as<FollowFusedSplitStepNode>()) {
    ApplySplitToScheduleSymbolic(stages, stage_to_axes, ps->stage_id, ps->iter_id,
                                 {ExtractFollowFusedSplitLength(ps, *split_ctx)},
                                 ps->factor_or_nparts);
  } else {
    StepApplyToSchedule(step, stages, stage_to_axes, schedule, transform_steps);
  }
}

PrimFunc RenameSplitVars(PrimFunc func, const Map<tir::Var, PrimExpr>& rename_map) {
  if (!rename_map.empty()) {
    Stmt body = Substitute(func->body, rename_map);
    func = PrimFunc(func->params, std::move(body), func->ret_type, func->buffer_map, func->attrs,
                    func->span);
  }
  return func;
}

PrimFunc WithUpdatedBody(const PrimFunc& func, Stmt body) {
  return PrimFunc(func->params, std::move(body), func->ret_type, func->buffer_map, func->attrs,
                  func->span);
}

PrimFunc LowerSymbolicPreVectorize(const ComputeDAG& dag, const State& state) {
  Array<te::Stage> stages;
  StageToAxesMap stage_to_axes;
  auto [schedule, tensors] = BuildInitialSchedule(dag, &stages, &stage_to_axes);

  SymbolicSplitContext split_ctx;
  for (size_t i = 0; i < state->transform_steps.size(); ++i) {
    StepApplyToScheduleSymbolic(state->transform_steps[i], static_cast<int>(i), &stages, &stage_to_axes,
                                &schedule, &split_ctx, state->transform_steps);
  }

  schedule = schedule.normalize();

  tvm::transform::PassContext pass_ctx = tvm::transform::PassContext::Current();
  bool debug_keep_trivial_loop =
      pass_ctx->GetConfig<Bool>("tir.debug_keep_trivial_loop", Bool(false)).value();

  tir::Stmt stmt =
      te::ScheduleOps(schedule, te::InferBound(schedule, &split_ctx.vcontext), debug_keep_trivial_loop);
  bool compact = te::VerifyCompactBuffer(stmt);

  std::unordered_map<te::Tensor, tir::Buffer> binds;
  Map<te::Tensor, tir::Buffer> out_binds;
  Array<ObjectRef> out_arg_list;
  Array<ObjectRef> args;
  for (const auto& tensor : tensors) {
    args.push_back(tensor);
  }
  GetBinds(args, compact, binds, &out_binds, &out_arg_list);

  PrimFunc func = te::SchedulePostProcToPrimFunc(out_arg_list, std::move(stmt), out_binds);
  func = WithAttr(std::move(func), "global_symbol", runtime::String("main"));
  func = WithAttr(std::move(func), "from_legacy_te_schedule", Bool(true));
  func = WithAttr(std::move(func), "tir.noalias", Bool(true));
  func = RenameSplitVars(std::move(func), split_ctx.rename_map);

  IRModule mod(Map<GlobalVar, BaseFunc>({{GlobalVar("main"), func}}));
  Array<tvm::transform::Pass> pass_list{
      tir::transform::InjectPrefetch(),
      tir::transform::StorageFlatten(64, false),
      tir::transform::NarrowDataType(32),
      tir::transform::Simplify(),
  };
  mod = tvm::transform::Sequential(pass_list)(std::move(mod));
  return Downcast<PrimFunc>((*mod->functions.begin()).second);
}

class VectorizedLoopCollector : public StmtExprVisitor {
 public:
  Array<PrimExpr> extents;

  void Collect(const Stmt& stmt) { this->VisitStmt(stmt); }

  void VisitStmt_(const ForNode* op) final {
    if (op->kind == ForKind::kVectorized) {
      extents.push_back(op->extent);
    }
    StmtExprVisitor::VisitStmt_(op);
  }
};

class VectorizedLoopExtentRewriter : public StmtExprMutator {
 public:
  explicit VectorizedLoopExtentRewriter(const Array<Integer>& replacements)
      : replacements_(replacements) {}

  PrimFunc Rewrite(const PrimFunc& func) {
    seen_ = 0;
    Stmt body = this->VisitStmt(func->body);
    ICHECK_EQ(seen_, replacements_.size())
        << "Expected " << replacements_.size() << " vectorized loops, saw " << seen_;
    return WithUpdatedBody(func, body);
  }

 private:
  Stmt VisitStmt_(const ForNode* op) final {
    For loop = Downcast<For>(StmtExprMutator::VisitStmt_(op));
    if (loop->kind != ForKind::kVectorized) {
      return loop;
    }

    ICHECK_LT(seen_, replacements_.size()) << "Too few replacement extents";
    PrimExpr new_extent = replacements_[seen_++];
    ForKind new_kind = loop->kind;
    if (const auto* imm = new_extent.as<IntImmNode>()) {
      if (imm->value <= 1) {
        new_kind = ForKind::kSerial;
      }
    }
    return For(loop->loop_var, loop->min, new_extent, new_kind, loop->body, loop->thread_binding,
               loop->annotations, loop->span);
  }

  Array<Integer> replacements_;
  size_t seen_{0};
};

class SharedBytesExtractor : public StmtExprVisitor {
 public:
  PrimExpr shared_bytes{0};

  void Collect(const Stmt& stmt) { this->VisitStmt(stmt); }

  void VisitStmt_(const tir::AllocateNode* op) final {
    StmtExprVisitor::VisitStmt_(op);
    auto scope = GetPtrStorageScope(op->buffer_var);
    runtime::StorageScope storage_scope = runtime::StorageScope::Create(scope);
    if (storage_scope.rank != runtime::StorageRank::kShared) {
      return;
    }

    PrimExpr alloc_count = 1;
    for (const auto& extent : op->extents) {
      alloc_count *= extent;
    }
    shared_bytes = shared_bytes + alloc_count * op->dtype.bytes() * op->dtype.lanes();
  }
};

class GpuCaseStatsCollector : public StmtExprVisitor {
 public:
  PrimExpr shared_bytes{Integer(0)};
  PrimExpr max_vthread{Integer(0)};
  Map<tir::Var, arith::IntSet> domains;
  int64_t max_vector_bytes{0};

  void Collect(const PrimFunc& func) {
    seen_thread_x_ = false;
    seen_thread_y_ = false;
    seen_thread_z_ = false;
    seen_vthread_ = false;
    thread_x_extent_ = Integer(1);
    thread_y_extent_ = Integer(1);
    thread_z_extent_ = Integer(1);
    vthread_extent_ = Integer(1);
    shared_bytes = Integer(0);
    max_vthread = Integer(0);
    domains = {};
    max_vector_bytes = 0;
    this->VisitStmt(func->body);
  }

  PrimExpr ThreadsPerBlock() const {
    arith::Analyzer analyzer;
    PrimExpr total = thread_x_extent_ * thread_y_extent_ * thread_z_extent_ * vthread_extent_;
    return analyzer.Simplify(total);
  }

  void VisitStmt_(const tir::AllocateNode* op) final {
    auto scope = GetPtrStorageScope(op->buffer_var);
    runtime::StorageScope storage_scope = runtime::StorageScope::Create(scope);
    if (storage_scope.rank == runtime::StorageRank::kShared) {
      PrimExpr alloc_count = 1;
      for (const auto& extent : op->extents) {
        alloc_count *= extent;
      }
      shared_bytes = shared_bytes + alloc_count * op->dtype.bytes() * op->dtype.lanes();
    }
    ObserveVectorBytes(op->dtype);
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const tir::AttrStmtNode* op) final {
    if (op->attr_key == tir::attr::thread_extent || op->attr_key == tir::attr::virtual_thread) {
      if (const auto* it = op->node.as<tir::IterVarNode>()) {
        std::string name = it->var->name_hint;
        if (name == "threadIdx.x") {
          MergeExtent(&thread_x_extent_, &seen_thread_x_, op->value);
        } else if (name == "threadIdx.y") {
          MergeExtent(&thread_y_extent_, &seen_thread_y_, op->value);
        } else if (name == "threadIdx.z") {
          MergeExtent(&thread_z_extent_, &seen_thread_z_, op->value);
        } else if (name == "vthread") {
          MergeExtent(&vthread_extent_, &seen_vthread_, op->value);
          max_vthread = tvm::max(max_vthread, op->value);
        }
        AddDomain(it->var, arith::IntSet::FromMinExtent(make_zero(op->value.dtype()), op->value));
      }
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const tir::ForNode* op) final {
    if (op->loop_var->name_hint == "vthread.s") {
      max_vthread = tvm::max(max_vthread, op->extent);
    }
    AddDomain(op->loop_var, arith::IntSet::FromMinExtent(op->min, op->extent));
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const tir::LetStmtNode* op) final {
    AddDomain(op->var, arith::IntSet::SinglePoint(op->value));
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitExpr_(const tir::LetNode* op) final {
    AddDomain(op->var, arith::IntSet::SinglePoint(op->value));
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const tir::CastNode* op) final {
    ObserveVectorBytes(op->dtype);
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const tir::BufferLoadNode* op) final {
    ObserveVectorBytes(op->dtype);
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const tir::BufferStoreNode* op) final {
    ObserveVectorBytes(op->value->dtype);
    StmtExprVisitor::VisitStmt_(op);
  }

 private:
  static void MergeExtent(PrimExpr* current, bool* seen, PrimExpr value) {
    arith::Analyzer analyzer;
    if (!*seen) {
      *current = analyzer.Simplify(value);
      *seen = true;
      return;
    }
    *current = analyzer.Simplify(tvm::max(*current, value));
  }

  void AddDomain(const tir::Var& var, const arith::IntSet& domain) {
    if (!domains.count(var)) {
      domains.Set(var, domain);
    }
  }

  void ObserveVectorBytes(DataType dtype) {
    if (dtype.lanes() <= 1) {
      return;
    }
    max_vector_bytes =
        std::max(max_vector_bytes, static_cast<int64_t>(dtype.lanes() * dtype.bytes()));
  }

  bool seen_thread_x_{false};
  bool seen_thread_y_{false};
  bool seen_thread_z_{false};
  bool seen_vthread_{false};
  PrimExpr thread_x_extent_{Integer(1)};
  PrimExpr thread_y_extent_{Integer(1)};
  PrimExpr thread_z_extent_{Integer(1)};
  PrimExpr vthread_extent_{Integer(1)};
};

tvm::transform::Pass MakePostVectorizePipeline() {
  Array<tvm::transform::Pass> pass_list{
      tir::transform::VectorizeLoop(true),
      tir::transform::InjectVirtualThread(),
      tir::transform::StorageRewrite(),
      tir::transform::Simplify(),
  };
  return tvm::transform::Sequential(pass_list);
}

int CountVectorizedLoops(const PrimFunc& pre_vectorize_func) {
  VectorizedLoopCollector collector;
  collector.Collect(pre_vectorize_func->body);
  return static_cast<int>(collector.extents.size());
}

PrimFunc LowerSymbolicPostVectorizeWithPipeline(const PrimFunc& pre_vectorize_func,
                                                const Array<Integer>& vector_case_values,
                                                int expected_vectorized_loops,
                                                const tvm::transform::Pass& pipeline) {
  ICHECK_EQ(expected_vectorized_loops, static_cast<int>(vector_case_values.size()))
      << "Expected " << expected_vectorized_loops << " vectorized loop extents, got "
      << vector_case_values.size();

  PrimFunc concretized = VectorizedLoopExtentRewriter(vector_case_values).Rewrite(pre_vectorize_func);
  IRModule mod(Map<GlobalVar, BaseFunc>({{GlobalVar("main"), concretized}}));
  mod = pipeline(std::move(mod));
  return Downcast<PrimFunc>((*mod->functions.begin()).second);
}

PrimFunc LowerSymbolicPostVectorize(const PrimFunc& pre_vectorize_func,
                                    const Array<Integer>& vector_case_values) {
  return LowerSymbolicPostVectorizeWithPipeline(
      pre_vectorize_func,
      vector_case_values,
      CountVectorizedLoops(pre_vectorize_func),
      MakePostVectorizePipeline());
}

class MaxVThreadExtractor : public StmtExprVisitor {
 public:
  PrimExpr max_vthread{0};

  void Collect(const Stmt& stmt) { this->VisitStmt(stmt); }

  void VisitStmt_(const tir::AttrStmtNode* op) final {
    if (op->attr_key == tir::attr::virtual_thread) {
      if (const auto* it = op->node.as<tir::IterVarNode>()) {
        if (it->var->name_hint == "vthread") {
          max_vthread = tvm::max(max_vthread, op->value);
        }
      }
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const tir::ForNode* op) final {
    if (op->loop_var->name_hint == "vthread.s") {
      max_vthread = tvm::max(max_vthread, op->extent);
    }
    StmtExprVisitor::VisitStmt_(op);
  }
};

class MaxThreadsPerBlockExtractor : public StmtExprVisitor {
 public:
  void Collect(const Stmt& stmt) {
    seen_thread_x_ = false;
    seen_thread_y_ = false;
    seen_thread_z_ = false;
    seen_vthread_ = false;
    thread_x_extent_ = Integer(1);
    thread_y_extent_ = Integer(1);
    thread_z_extent_ = Integer(1);
    vthread_extent_ = Integer(1);
    this->VisitStmt(stmt);
  }

  PrimExpr ThreadsPerBlock() const {
    arith::Analyzer analyzer;
    PrimExpr total = thread_x_extent_ * thread_y_extent_ * thread_z_extent_ * vthread_extent_;
    return analyzer.Simplify(total);
  }

  void VisitStmt_(const tir::AttrStmtNode* op) final {
    if (op->attr_key == tir::attr::thread_extent || op->attr_key == tir::attr::virtual_thread) {
      if (const auto* it = op->node.as<tir::IterVarNode>()) {
        std::string name = it->var->name_hint;
        if (name == "threadIdx.x") {
          MergeExtent(&thread_x_extent_, &seen_thread_x_, op->value);
        } else if (name == "threadIdx.y") {
          MergeExtent(&thread_y_extent_, &seen_thread_y_, op->value);
        } else if (name == "threadIdx.z") {
          MergeExtent(&thread_z_extent_, &seen_thread_z_, op->value);
        } else if (name == "vthread") {
          MergeExtent(&vthread_extent_, &seen_vthread_, op->value);
        }
      }
    }
    StmtExprVisitor::VisitStmt_(op);
  }

 private:
  static void MergeExtent(PrimExpr* current, bool* seen, PrimExpr value) {
    arith::Analyzer analyzer;
    if (!*seen) {
      *current = analyzer.Simplify(value);
      *seen = true;
      return;
    }
    *current = analyzer.Simplify(tvm::max(*current, value));
  }

  bool seen_thread_x_{false};
  bool seen_thread_y_{false};
  bool seen_thread_z_{false};
  bool seen_vthread_{false};
  PrimExpr thread_x_extent_{Integer(1)};
  PrimExpr thread_y_extent_{Integer(1)};
  PrimExpr thread_z_extent_{Integer(1)};
  PrimExpr vthread_extent_{Integer(1)};
};

class MaxVectorBytesExtractor : public StmtExprVisitor {
 public:
  int64_t max_vector_bytes{0};

  int64_t Collect(const PrimFunc& func) {
    max_vector_bytes = 0;
    this->VisitStmt(func->body);
    return max_vector_bytes;
  }

  void VisitStmt_(const tir::AllocateNode* op) final {
    Observe(op->dtype);
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitExpr_(const tir::CastNode* op) final {
    Observe(op->dtype);
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const tir::BufferLoadNode* op) final {
    Observe(op->dtype);
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const tir::BufferStoreNode* op) final {
    Observe(op->value->dtype);
    StmtExprVisitor::VisitStmt_(op);
  }

 private:
  void Observe(DataType dtype) {
    if (dtype.lanes() <= 1) {
      return;
    }
    max_vector_bytes =
        std::max(max_vector_bytes, static_cast<int64_t>(dtype.lanes() * dtype.bytes()));
  }
};

class VectorBytesVerifier : public StmtExprVisitor {
 public:
  bool Verify(const PrimFunc& func, int max_vector_bytes) {
    max_vector_bytes_ = static_cast<size_t>(max_vector_bytes);
    errors_.clear();
    this->VisitStmt(func->body);
    return errors_.empty();
  }

  void VisitStmt_(const tir::AllocateNode* op) final {
    if (op->dtype.lanes() > 1) {
      CheckVectorBytes(op->dtype);
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitExpr_(const tir::CastNode* op) final {
    if (op->dtype.lanes() > 1) {
      CheckVectorBytes(op->dtype);
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const tir::BufferLoadNode* op) final {
    if (op->dtype.lanes() > 1) {
      CheckVectorBytes(op->dtype);
      CheckBufferIndicesVectorizable(op->indices);
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const tir::BufferStoreNode* op) final {
    if (op->value->dtype.lanes() > 1) {
      CheckVectorBytes(op->value->dtype);
      CheckBufferIndicesVectorizable(op->indices);
    }
    StmtExprVisitor::VisitStmt_(op);
  }

 private:
  void CheckVectorBytes(DataType dtype) {
    if (static_cast<size_t>(dtype.lanes() * dtype.bytes()) > max_vector_bytes_) {
      errors_.push_back(true);
    }
  }

  void CheckBufferIndicesVectorizable(const Array<PrimExpr>& indices) {
    for (const auto& index : indices) {
      if (const auto* ramp = index.as<tir::RampNode>()) {
        if (!is_one(ramp->stride) &&
            static_cast<size_t>(ramp->dtype.lanes() * ramp->dtype.bytes()) >
                max_vector_bytes_) {
          errors_.push_back(true);
        }
      }
    }
  }

  size_t max_vector_bytes_{0};
  std::vector<bool> errors_;
};

class RuntimeDomainCollector : public StmtExprVisitor {
 public:
  Map<tir::Var, arith::IntSet> domains;

  void Collect(const Stmt& stmt) { this->VisitStmt(stmt); }

  void VisitStmt_(const tir::AttrStmtNode* op) final {
    if ((op->attr_key == tir::attr::thread_extent || op->attr_key == tir::attr::virtual_thread) &&
        op->node.as<tir::IterVarNode>()) {
      tir::IterVar iter_var = Downcast<tir::IterVar>(op->node);
      AddDomain(iter_var->var, arith::IntSet::FromMinExtent(make_zero(op->value.dtype()), op->value));
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const tir::ForNode* op) final {
    AddDomain(op->loop_var, arith::IntSet::FromMinExtent(op->min, op->extent));
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const tir::LetStmtNode* op) final {
    AddDomain(op->var, arith::IntSet::SinglePoint(op->value));
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitExpr_(const tir::LetNode* op) final {
    AddDomain(op->var, arith::IntSet::SinglePoint(op->value));
    StmtExprVisitor::VisitExpr_(op);
  }

 private:
  void AddDomain(const tir::Var& var, const arith::IntSet& domain) {
    if (!domains.count(var)) {
      domains.Set(var, domain);
    }
  }
};

PrimExpr UpperBoundOverRuntimeVars(const PrimExpr& expr, const Map<tir::Var, arith::IntSet>& domains) {
  auto is_symbolic_inf = [](const PrimExpr& value) {
    if (const auto* var = value.as<tir::VarNode>()) {
      return var->name_hint == "pos_inf" || var->name_hint == "neg_inf";
    }
    return false;
  };

  arith::Analyzer analyzer;
  PrimExpr simplified = analyzer.Simplify(expr);
  if (domains.empty()) {
    return simplified;
  }

  PrimExpr upper = analyzer.Simplify(analyzer.int_set(expr, domains).max());
  if (!is_symbolic_inf(upper)) {
    return upper;
  }

  Map<tir::Var, PrimExpr> subst;
  for (const auto& kv : domains) {
    PrimExpr max_value = analyzer.Simplify(kv.second.max());
    if (is_symbolic_inf(max_value)) {
      continue;
    }
    subst.Set(kv.first, max_value);
  }

  if (!subst.empty()) {
    PrimExpr substituted = analyzer.Simplify(tir::Substitute(expr, subst));
    if (!is_symbolic_inf(substituted)) {
      return substituted;
    }
  }

  return simplified;
}

bool VerifyVectorOnly(const PrimFunc& func, int max_vector_bytes) {
  return VectorBytesVerifier().Verify(func, max_vector_bytes);
}

Array<ObjectRef> BuildGpuCaseStats(const PrimFunc& post_vectorize_func) {
  arith::Analyzer analyzer;
  GpuCaseStatsCollector collector;
  collector.Collect(post_vectorize_func);

  PrimExpr shared_upper =
      UpperBoundOverRuntimeVars(collector.shared_bytes, collector.domains);
  PrimExpr vthread_upper =
      UpperBoundOverRuntimeVars(collector.max_vthread, collector.domains);
  PrimExpr max_threads_upper =
      UpperBoundOverRuntimeVars(collector.ThreadsPerBlock(), collector.domains);

  return {analyzer.Simplify(shared_upper), analyzer.Simplify(vthread_upper),
          analyzer.Simplify(max_threads_upper), Integer(collector.max_vector_bytes)};
}

Array<ObjectRef> ExtractGpuCaseStats(const PrimFunc& pre_vectorize_func,
                                     const Array<Integer>& vector_case_values) {
  PrimFunc post = LowerSymbolicPostVectorize(pre_vectorize_func, vector_case_values);
  return BuildGpuCaseStats(post);
}

Array<Array<ObjectRef>> ExtractAllGpuCaseStats(
    const PrimFunc& pre_vectorize_func, const Array<Array<Integer>>& vector_case_values_list) {
  Array<Array<ObjectRef>> results;
  int expected_vectorized_loops = CountVectorizedLoops(pre_vectorize_func);
  tvm::transform::Pass pipeline = MakePostVectorizePipeline();

  for (const auto& vector_case_values : vector_case_values_list) {
    PrimFunc post = LowerSymbolicPostVectorizeWithPipeline(
        pre_vectorize_func, vector_case_values, expected_vectorized_loops, pipeline);
    results.push_back(Downcast<Array<ObjectRef>>(BuildGpuCaseStats(post)));
  }
  return results;
}

}  // namespace

TVM_REGISTER_GLOBAL("constrained_gen.lower_symbolic_pre_vectorize")
    .set_body_typed([](ComputeDAG dag, State state) { return LowerSymbolicPreVectorize(dag, state); });

TVM_REGISTER_GLOBAL("constrained_gen.list_vectorized_loop_extents").set_body_typed([](PrimFunc func) {
  VectorizedLoopCollector collector;
  collector.Collect(func->body);
  return collector.extents;
});

TVM_REGISTER_GLOBAL("constrained_gen.lower_symbolic_post_vectorize")
    .set_body_typed([](PrimFunc pre_vectorize_func, Array<Integer> vector_case_values) {
      return LowerSymbolicPostVectorize(pre_vectorize_func, vector_case_values);
    });

TVM_REGISTER_GLOBAL("constrained_gen.extract_gpu_case_stats")
    .set_body_typed([](PrimFunc pre_vectorize_func, Array<Integer> vector_case_values) {
      return ExtractGpuCaseStats(pre_vectorize_func, vector_case_values);
    });

TVM_REGISTER_GLOBAL("constrained_gen.extract_all_gpu_case_stats")
    .set_body_typed([](PrimFunc pre_vectorize_func, Array<Array<Integer>> vector_case_values_list) {
      return ExtractAllGpuCaseStats(pre_vectorize_func, vector_case_values_list);
    });

}  // namespace auto_scheduler
}  // namespace tvm
