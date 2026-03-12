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

#include <tvm/arith/analyzer.h>
#include <tvm/arith/var_context.h>
#include <tvm/tir/op.h>

namespace tvm {
namespace arith {

Array<tir::SizeVar> VarContext::GetSplitVars(PrimExpr extent, size_t n_splits, bool whole_div) {
  static_cast<void>(extent);
  static_cast<void>(whole_div);

  std::string group_idx = std::to_string(split_counter_++);
  Array<tir::SizeVar> vars;
  for (size_t i = 0; i < n_splits; ++i) {
    vars.push_back(tir::SizeVar("sp_" + group_idx + "_" + std::to_string(i)));
  }
  return vars;
}

std::pair<PrimExpr, PrimExpr> VarContext::GetSplitSizes(const PrimExpr& extent, PrimExpr factor,
                                                        bool no_tighten_factor) const {
  arith::Analyzer analyzer;
  PrimExpr min_factor = no_tighten_factor ? factor : tvm::min(extent, factor);
  PrimExpr divided = indexdiv(extent + (factor - 1), factor);
  return {analyzer.Simplify(min_factor), analyzer.Simplify(divided)};
}

PrimExpr VarContext::DefineConstShorthand(PrimExpr expr) const { return expr; }

}  // namespace arith
}  // namespace tvm
