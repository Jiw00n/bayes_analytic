"""
full_generator — Constrained Symbolic Schedule Generator for TVM Ansor.

모듈 구성:
  sym_types          : SymExpr, SymIter, SymStage, eval_sym_extent, 상수
  symbolic_state     : SymbolicState (순수 상태 객체)
  transform_applier  : TransformApplier (transform step 적용)
  param_manager      : SymParamManager, build_symbolic_state, verify_symbolic_state
  record_loader      : sketch fingerprint, 레코드 로드 / 그룹핑
  expr_nodes         : ExprNode 트리 + _parse_expr_tree
  schedule_generator : ScheduleGenerator (HW 제약 기반 파라미터 생성)
  tvm_verify         : TVM API 검증 유틸리티
"""

from .sym_types import (
    SymExpr, SymIter, SymStage, eval_sym_extent,
    ANNOTATION_STR, CA_ROOT, CA_INLINED, CA_ITER,
)
from .symbolic_state import SymbolicState
from .transform_applier import TransformApplier
from .param_manager import SymParamManager, build_symbolic_state, verify_symbolic_state
from .record_loader import (
    state_sketch_fingerprint,
    load_records_from_dir,
    group_records_by_wkey_and_sketch,
    group_by_sketches_from_json,
)
from .expr_nodes import (
    ExprNode, ConstNode, VarNode, MulNode, AddNode, SubNode,
    MinNode, CeilDivNode, ScaleMulNode, SumNode,
    parse_expr_tree,
)
from .schedule_generator import ScheduleGenerator
from .tvm_verify import (
    lower_with_gpu_passes, verify_gpu_module, params_to_state,
    GPU_PASSES, GPU_VERIFY_CONSTRAINTS,
)
