"""
sym_types — 기본 심볼릭 타입: SymExpr, SymIter, SymStage, eval_sym_extent, 상수 정의.
"""
import math

builtins_min = min
builtins_max = max

# ── Annotation 문자열 매핑 (C++ IteratorAnnotationString 동일) ──
ANNOTATION_STR = {
    0: "for",           # kNone
    1: "unroll",        # kUnroll
    2: "vectorize",     # kVectorize
    3: "parallel",      # kParallel
    4: "vthread",       # kVThread
    5: "blockIdx.x",    # kBlockX
    6: "threadIdx.x",   # kThreadX
    7: "blockIdx.y",    # kBlockY
    8: "threadIdx.y",   # kThreadY
    9: "blockIdx.z",    # kBlockZ
    10: "threadIdx.z",  # kThreadZ
    11: "tensorize",    # kTensorize
}

# ── ComputeAtKind ──
CA_ROOT    = 0  # kRoot
CA_INLINED = 1  # kInlined
CA_ITER    = 2  # kIter


class SymExpr:
    """
    Symbolic expression wrapper.
    실제 값(int)이면 그냥 int, symbolic이면 문자열.
    연산(ceil div, mul 등)을 문자열로 합성.
    """
    def __init__(self, val):
        """값(정수 또는 심볼릭 문자열)으로 SymExpr를 만든다."""
        self.val = val

    @property
    def is_concrete(self):
        """값이 정수면 True, 심볼릭이면 False."""
        return isinstance(self.val, int)

    def __repr__(self):
        return str(self.val) if self.val is not None else "None"

    def __str__(self):
        return str(self.val) if self.val is not None else "None"

    def __int__(self):
        if self.is_concrete:
            return self.val
        raise ValueError(f"Cannot convert symbolic '{self.val}' to int")

    @staticmethod
    def ceildiv(a, b):
        """올림 나눗셈; 둘 다 정수면 정수, 아니면 ceil(a/b) 심볼 문자열."""
        if isinstance(a, SymExpr): a = a.val
        if isinstance(b, SymExpr): b = b.val
        if isinstance(a, int) and isinstance(b, int):
            return SymExpr((a + b - 1) // b)
        a_str = str(a)
        b_str = str(b)
        if SymExpr._needs_parens_for_mul(a_str):
            a_str = f"({a_str})"
        return SymExpr(f"ceil({a_str}/({b_str}))")

    @staticmethod
    def _needs_parens_for_mul(s):
        """문자열 expression이 mul에서 사용될 때 괄호가 필요한지 판단."""
        if not isinstance(s, str):
            return False
        depth = 0
        i = 0
        while i < len(s):
            c = s[i]
            if c == '(':
                depth += 1
            elif c == ')':
                depth -= 1
            elif depth == 0 and c == '+':
                return True
            elif depth == 0 and c == '-' and i > 0 and s[i-1] == ' ':
                return True
            i += 1
        return False

    @staticmethod
    def mul(a, b):
        """곱셈; 둘 다 정수면 정수, 아니면 심볼릭 곱 문자열."""
        if isinstance(a, SymExpr): a = a.val
        if isinstance(b, SymExpr): b = b.val
        if isinstance(a, int) and isinstance(b, int):
            return SymExpr(a * b)
        if a == 1: return SymExpr(b)
        if b == 1: return SymExpr(a)
        a_str = str(a)
        b_str = str(b)
        if SymExpr._needs_parens_for_mul(a_str):
            a_str = f"({a_str})"
        if SymExpr._needs_parens_for_mul(b_str):
            b_str = f"({b_str})"
        return SymExpr(f"{a_str}*{b_str}")

    @staticmethod
    def product(items):
        """여러 항목을 순서대로 곱한 SymExpr를 반환한다."""
        result = SymExpr(1)
        for item in items:
            result = SymExpr.mul(result, item)
        return result

    @staticmethod
    def min(a, b):
        """두 값의 최소; 둘 다 정수면 정수, 아니면 min(...,...) 심볼 문자열."""
        if a is None or b is None:
            return a if b is None else b
        if isinstance(a, SymExpr): a_val = a.val
        else: a_val = a
        if isinstance(b, SymExpr): b_val = b.val
        else: b_val = b
        if isinstance(a_val, int) and isinstance(b_val, int):
            return SymExpr(builtins_min(a_val, b_val))
        return SymExpr(f"min({a_val},{b_val})")

    @staticmethod
    def max(items):
        """여러 항목의 최대; 전부 정수면 정수, 아니면 max(...,...) 심볼 문자열."""
        vals = []
        for item in items:
            if item is None:
                continue
            vals.append(item.val if isinstance(item, SymExpr) else item)
        if not vals:
            return SymExpr(0)
        if len(vals) == 1:
            return SymExpr(vals[0])
        if all(isinstance(val, int) for val in vals):
            return SymExpr(builtins_max(vals))
        return SymExpr("max(" + ",".join(str(val) for val in vals) + ")")


class SymIter:
    """Iterator (C++ Iterator 대응)"""
    def __init__(self, name, extent, annotation=0, iter_kind=0, min_value=None):
        """심볼릭 이터레이터(이름, min, extent, 어노테이션, iter_kind)를 만든다."""
        self.name = name
        self.extent = extent         # SymExpr or None
        self.min_value = SymExpr(0) if min_value is None else min_value
        self.annotation = annotation # int
        self.iter_kind = iter_kind   # int

    def clone(self):
        """이 SymIter의 복사본을 반환한다."""
        return SymIter(self.name,
                       SymExpr(self.extent.val) if self.extent else None,
                       self.annotation, self.iter_kind,
                       min_value=SymExpr(self.min_value.val) if self.min_value else None)

    def __repr__(self):
        ann = ANNOTATION_STR.get(self.annotation, "?")
        if self.extent is not None:
            return f"{ann} {self.name} ({self.min_value},{self.extent})"
        else:
            return f"{ann} {self.name} (None)"


class SymStage:
    """Stage (C++ Stage 대응)"""
    def __init__(self, op_name, op_type, iters, compute_at=CA_ROOT,
                 auto_unroll_max_step=None, storage_offset=0, dtype="float32"):
        """심볼릭 스테이지(op명, 타입, 이터 목록, compute_at 등)를 만든다."""
        self.op_name = op_name
        self.op_type = op_type
        self.iters = list(iters)
        self.compute_at = compute_at
        self.auto_unroll_max_step = auto_unroll_max_step
        self.storage_offset = storage_offset
        self.attach_stage_id = None
        self.attach_iter_id = None
        self.dtype = dtype

    def clone(self):
        """이 SymStage의 복사본을 반환한다."""
        cloned = SymStage(
            self.op_name,
            self.op_type,
            [it.clone() for it in self.iters],
            compute_at=self.compute_at,
            auto_unroll_max_step=SymExpr(self.auto_unroll_max_step.val)
            if self.auto_unroll_max_step is not None
            else None,
            storage_offset=self.storage_offset,
            dtype=self.dtype,
        )
        cloned.attach_stage_id = self.attach_stage_id
        cloned.attach_iter_id = self.attach_iter_id
        return cloned

    @property
    def dtype_bytes(self):
        """dtype의 바이트 수를 반환."""
        import tvm
        return tvm.DataType(self.dtype).bits // 8


def eval_sym_extent(expr, sym_map):
    """심볼 extent 문자열을 sym_map으로 치환한 뒤 계산해 정수 또는 실패 시 문자열을 반환한다."""
    if expr is None:
        return None
    s_val = str(expr)
    if s_val == "None":
        return None
    try:
        return int(s_val)
    except ValueError:
        pass
    evaluated = s_val
    evaluated = evaluated.replace("ceil(", "math.ceil(")
    for sym_name in sorted(sym_map.keys(), key=len, reverse=True):
        if sym_map[sym_name] is not None:
            evaluated = evaluated.replace(sym_name, str(sym_map[sym_name]))
    try:
        return int(eval(evaluated))
    except Exception:
        return f"EVAL_FAIL({s_val}→{evaluated})"
