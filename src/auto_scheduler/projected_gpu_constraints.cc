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

#include <tvm/arith/var_context.h>
#include <tvm/auto_scheduler/compute_dag.h>
#include <tvm/auto_scheduler/loop_state.h>
#include <tvm/auto_scheduler/search_task.h>
#include <tvm/auto_scheduler/transform_step.h>
#include <tvm/ir/module.h>
#include <tvm/runtime/registry.h>
#include <tvm/te/operation.h>
#include <tvm/te/schedule_pass.h>
#include <tvm/tir/function.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <string>
#include <unordered_map>
#include <utility>

namespace tvm {
namespace auto_scheduler {
namespace {

using tir::For;
using tir::ForKind;
using tir::IterVar;
using tir::PrimFunc;
using tir::Stmt;
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

}  // namespace

TVM_REGISTER_GLOBAL("constrained_gen.lower_symbolic_pre_vectorize")
    .set_body_typed([](ComputeDAG dag, State state) { return LowerSymbolicPreVectorize(dag, state); });

TVM_REGISTER_GLOBAL("constrained_gen.list_vectorized_loop_extents").set_body_typed([](PrimFunc func) {
  VectorizedLoopCollector collector;
  collector.Collect(func->body);
  return collector.extents;
});

}  // namespace auto_scheduler
}  // namespace tvm
