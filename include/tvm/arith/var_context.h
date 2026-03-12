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

/*!
 * \file tvm/arith/var_context.h
 * \brief Minimal symbolic split-variable context used by constrained_gen.
 */

#ifndef TVM_ARITH_VAR_CONTEXT_H_
#define TVM_ARITH_VAR_CONTEXT_H_

#include <tvm/tir/expr.h>
#include <tvm/tir/var.h>

#include <utility>

namespace tvm {
namespace arith {

/*!
 * \brief A lightweight context for creating symbolic split knobs and deriving
 * split bounds during TE InferBound.
 */
class VarContext {
 public:
  VarContext() = default;

  /*!
   * \brief Create symbolic split knobs for one split step.
   * \param extent The original iterator extent. Kept for API compatibility.
   * \param n_splits The number of explicit split factors.
   * \param whole_div Whether the split is whole-divisible. Kept for API compatibility.
   */
  Array<tir::SizeVar> GetSplitVars(PrimExpr extent, size_t n_splits, bool whole_div = true);

  /*!
   * \brief Derive the child and parent extents of a split relation.
   * \param extent The parent extent.
   * \param factor The explicit split factor or nparts.
   * \param no_tighten_factor Whether min(extent, factor) tightening is disabled.
   */
  std::pair<PrimExpr, PrimExpr> GetSplitSizes(const PrimExpr& extent, PrimExpr factor,
                                              bool no_tighten_factor = false) const;

  /*!
   * \brief Define shorthand for a long constant expression.
   * \note The constrained_gen path keeps expressions expanded, so this is a no-op.
   */
  PrimExpr DefineConstShorthand(PrimExpr expr) const;

 private:
  size_t split_counter_{0};
};

}  // namespace arith
}  // namespace tvm

#endif  // TVM_ARITH_VAR_CONTEXT_H_
