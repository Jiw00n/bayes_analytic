"""
expr_nodes — ExprNode 트리 클래스 + parse_expr_tree 파서.

제약식 LHS를 나타내는 표현식 트리 노드.
각 노드는 partial assignment 상태에서 (lo, hi) 구간을 반환할 수 있다.
"""

builtins_min = min
from itertools import product


class ExprNode:
    """제약식 LHS를 나타내는 표현식 트리 노드 (기본 클래스)."""
    pass


class ConstNode(ExprNode):
    """정수 상수."""
    __slots__ = ('val',)
    def __init__(self, val):
        self.val = int(val)

    def interval(self, domains):
        return (self.val, self.val)

    def evaluate(self, assignment):
        return self.val

    def variables(self):
        return set()

    def __repr__(self):
        return str(self.val)


class VarNode(ExprNode):
    """심볼릭 변수 (sp_X_Y)."""
    __slots__ = ('name',)
    def __init__(self, name):
        self.name = name

    def interval(self, domains):
        dom = domains.get(self.name)
        if dom is None:
            return (1, 1)
        if isinstance(dom, int):
            return (dom, dom)
        return (dom[0], dom[-1])

    def evaluate(self, assignment):
        return assignment.get(self.name, 1)

    def variables(self):
        return {self.name}

    def __repr__(self):
        return self.name


class MulNode(ExprNode):
    """곱셈: left * right."""
    __slots__ = ('left', 'right')
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def interval(self, domains):
        a_lo, a_hi = self.left.interval(domains)
        b_lo, b_hi = self.right.interval(domains)
        return (a_lo * b_lo, a_hi * b_hi)

    def evaluate(self, assignment):
        return self.left.evaluate(assignment) * self.right.evaluate(assignment)

    def variables(self):
        return self.left.variables() | self.right.variables()

    def __repr__(self):
        return f"({self.left}*{self.right})"


class AddNode(ExprNode):
    """덧셈: left + right."""
    __slots__ = ('left', 'right')
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def interval(self, domains):
        a_lo, a_hi = self.left.interval(domains)
        b_lo, b_hi = self.right.interval(domains)
        return (a_lo + b_lo, a_hi + b_hi)

    def evaluate(self, assignment):
        return self.left.evaluate(assignment) + self.right.evaluate(assignment)

    def variables(self):
        return self.left.variables() | self.right.variables()

    def __repr__(self):
        return f"({self.left}+{self.right})"


class SubNode(ExprNode):
    """뺄셈: left - right."""
    __slots__ = ('left', 'right')
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def interval(self, domains):
        a_lo, a_hi = self.left.interval(domains)
        b_lo, b_hi = self.right.interval(domains)
        return (a_lo - b_hi, a_hi - b_lo)

    def evaluate(self, assignment):
        return self.left.evaluate(assignment) - self.right.evaluate(assignment)

    def variables(self):
        return self.left.variables() | self.right.variables()

    def __repr__(self):
        return f"({self.left}-{self.right})"


class MinNode(ExprNode):
    """min(left, right)."""
    __slots__ = ('left', 'right')
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def interval(self, domains):
        a_lo, a_hi = self.left.interval(domains)
        b_lo, b_hi = self.right.interval(domains)
        return (builtins_min(a_lo, b_lo), builtins_min(a_hi, b_hi))

    def evaluate(self, assignment):
        return builtins_min(self.left.evaluate(assignment), self.right.evaluate(assignment))

    def variables(self):
        return self.left.variables() | self.right.variables()

    def __repr__(self):
        return f"min({self.left},{self.right})"


class CeilDivNode(ExprNode):
    """ceil(left / right)."""
    __slots__ = ('left', 'right')
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def interval(self, domains):
        a_lo, a_hi = self.left.interval(domains)
        b_lo, b_hi = self.right.interval(domains)
        b_lo = max(b_lo, 1)
        b_hi = max(b_hi, 1)
        lo = (a_lo + b_hi - 1) // b_hi
        hi = (a_hi + b_lo - 1) // b_lo
        return (max(lo, 1), max(hi, 1))

    def evaluate(self, assignment):
        a = self.left.evaluate(assignment)
        b = self.right.evaluate(assignment)
        b = max(b, 1)
        return (a + b - 1) // b

    def variables(self):
        return self.left.variables() | self.right.variables()

    def __repr__(self):
        return f"ceil({self.left}/{self.right})"


class ScaleMulNode(ExprNode):
    """child * constant (dtype_bytes 등 곱)."""
    __slots__ = ('child', 'scale')
    def __init__(self, child, scale):
        self.child = child
        self.scale = scale

    def interval(self, domains):
        lo, hi = self.child.interval(domains)
        return (lo * self.scale, hi * self.scale)

    def evaluate(self, assignment):
        return self.child.evaluate(assignment) * self.scale

    def variables(self):
        return self.child.variables()

    def __repr__(self):
        return f"({self.child}*{self.scale})"


class SumNode(ExprNode):
    """여러 자식의 합. shared memory 총합에 사용."""
    __slots__ = ('children',)
    def __init__(self, children):
        self.children = list(children)

    def interval(self, domains):
        lo = sum(c.interval(domains)[0] for c in self.children)
        hi = sum(c.interval(domains)[1] for c in self.children)
        return (lo, hi)

    def evaluate(self, assignment):
        return sum(c.evaluate(assignment) for c in self.children)

    def variables(self):
        v = set()
        for c in self.children:
            v |= c.variables()
        return v

    def __repr__(self):
        return " + ".join(str(c) for c in self.children)


class MaxNode(ExprNode):
    """여러 자식의 최댓값."""

    __slots__ = ("children",)

    def __init__(self, children):
        self.children = list(children)

    def interval(self, domains):
        if not self.children:
            return (0, 0)
        intervals = [child.interval(domains) for child in self.children]
        return (
            max(interval[0] for interval in intervals),
            max(interval[1] for interval in intervals),
        )

    def evaluate(self, assignment):
        if not self.children:
            return 0
        return max(child.evaluate(assignment) for child in self.children)

    def variables(self):
        vars_in = set()
        for child in self.children:
            vars_in |= child.variables()
        return vars_in

    def __repr__(self):
        if not self.children:
            return "0"
        return "max(" + ", ".join(str(child) for child in self.children) + ")"


def _safe_int_expr(expr):
    try:
        return int(expr)
    except TypeError as err:
        raise TypeError(f"Expected concrete integer expression, got {expr}") from err


def _maybe_int_expr(expr):
    try:
        return int(expr)
    except TypeError:
        return None


class PrimExprNode(ExprNode):
    """TVM PrimExpr wrapper."""

    __slots__ = ("expr", "_var_map")

    def __init__(self, expr):
        self.expr = expr
        self._var_map = {}
        self._rebuild_var_map()

    def _rebuild_var_map(self):
        import tvm

        var_map = {}

        def visit(node):
            if isinstance(node, tvm.tir.Var):
                var_map.setdefault(str(node.name), node)

        tvm.tir.stmt_functor.post_order_visit(self.expr, visit)
        self._var_map = var_map

    def __getstate__(self):
        return {"expr": self.expr}

    def __setstate__(self, state):
        self.expr = state["expr"]
        self._var_map = {}
        self._rebuild_var_map()

    def interval(self, domains):
        import tvm

        analyzer = tvm.arith.Analyzer()
        if not self._var_map:
            val = _safe_int_expr(analyzer.simplify(self.expr))
            return (val, val)

        dom_map = {}
        for name, var in self._var_map.items():
            dom = domains.get(name)
            if dom is None:
                lo = hi = 1
            elif isinstance(dom, int):
                lo = hi = dom
            else:
                lo, hi = int(dom[0]), int(dom[-1])
            dom_map[var] = tvm.arith.IntervalSet(lo, hi)

        interval = analyzer.int_set(self.expr, dom_map)
        lo = _maybe_int_expr(analyzer.simplify(interval.min_value))
        hi = _maybe_int_expr(analyzer.simplify(interval.max_value))
        if lo is None or hi is None:
            return (0, 1 << 60)
        return (lo, hi)

    def evaluate(self, assignment):
        import tvm

        expr = self.expr
        if self._var_map:
            subst = {var: assignment.get(name, 1) for name, var in self._var_map.items()}
            expr = tvm.tir.stmt_functor.substitute(expr, subst)
        return _safe_int_expr(tvm.arith.Analyzer().simplify(expr))

    def variables(self):
        return set(self._var_map.keys())

    def __repr__(self):
        return str(self.expr)


class ProjectedExprNode(ExprNode):
    """Display/variables use projected expression, evaluation uses exact expression."""

    __slots__ = ("display", "exact")

    def __init__(self, display, exact):
        self.display = display
        self.exact = exact

    def interval(self, domains):
        return self.display.interval(domains)

    def evaluate(self, assignment):
        return self.exact.evaluate(assignment)

    def variables(self):
        return self.display.variables()

    def __repr__(self):
        return str(self.display)


class BoundedMaxNode(ExprNode):
    """Exact max over bounded auxiliary vars."""

    __slots__ = ("child", "bound_vars")

    def __init__(self, child, bound_vars):
        self.child = child
        self.bound_vars = dict(bound_vars)

    def interval(self, domains):
        return self.child.interval(domains)

    def evaluate(self, assignment):
        return self.child.evaluate(assignment)

    def variables(self):
        vars_in = set(self.child.variables()) - set(self.bound_vars.keys())
        for lo_expr, hi_expr in self.bound_vars.values():
            vars_in |= lo_expr.variables()
            vars_in |= hi_expr.variables()
        return vars_in

    def __repr__(self):
        if not self.bound_vars:
            return str(self.child)
        bounds = ", ".join(
            f"{name} in [{lo_expr}, {hi_expr}]"
            for name, (lo_expr, hi_expr) in self.bound_vars.items()
        )
        return f"max_{{{bounds}}}({self.child})"


class CaseSplitNode(ExprNode):
    """Selector equality case table over child ExprNode values."""

    __slots__ = ("selectors", "cases", "default", "_case_map", "extra_domains")

    def __init__(self, selectors, cases, default=None, extra_domains=None):
        self.selectors = list(selectors)
        self.cases = [
            {"values": tuple(case["values"]), "expr": case["expr"]}
            for case in cases
        ]
        self.default = default if default is not None else ConstNode(1 << 60)
        self._case_map = {case["values"]: case["expr"] for case in self.cases}
        self.extra_domains = {}
        if extra_domains:
            for name, bounds in extra_domains.items():
                lo, hi = bounds
                if not isinstance(lo, ExprNode):
                    lo = ConstNode(lo)
                if not isinstance(hi, ExprNode):
                    hi = ConstNode(hi)
                self.extra_domains[name] = (lo, hi)

    def _augment_domains(self, domains):
        full = dict(domains)
        for name, (lo_expr, hi_expr) in self.extra_domains.items():
            if name in full:
                continue
            lo_vars = lo_expr.variables()
            hi_vars = hi_expr.variables()
            if all(not isinstance(full.get(var), list) for var in lo_vars):
                lo = lo_expr.evaluate(full)
            else:
                lo = lo_expr.interval(full)[0]
            if all(not isinstance(full.get(var), list) for var in hi_vars):
                hi = hi_expr.evaluate(full)
            else:
                hi = hi_expr.interval(full)[1]
            if hi < lo:
                hi = lo
            full[name] = [lo, hi]
        return full

    def _case_feasible(self, values, domains):
        for selector, target in zip(self.selectors, values):
            lo, hi = selector.interval(domains)
            if target < lo or target > hi:
                return False
        return True

    def feasible_case_values(self, domains):
        if not self.cases:
            return []
        if not self.selectors:
            return [case["values"] for case in self.cases]
        return [
            case["values"]
            for case in self.cases
            if self._case_feasible(case["values"], domains)
        ]

    def interval_with_feasible_cases(self, domains, feasible_case_values):
        if not self.cases:
            return self.default.interval(domains)

        if not self.selectors:
            expr = self._case_map.get(tuple())
            if expr is not None:
                return expr.interval(domains)
            intervals = [case["expr"].interval(domains) for case in self.cases]
        else:
            intervals = [
                self._case_map[values].interval(domains)
                for values in feasible_case_values
                if values in self._case_map
            ]

        if not intervals:
            return self.default.interval(domains)
        lo = builtins_min(interval[0] for interval in intervals)
        hi = max(interval[1] for interval in intervals)
        return (lo, hi)

    def interval(self, domains):
        domains = self._augment_domains(domains)
        feasible_case_values = self.feasible_case_values(domains)
        return self.interval_with_feasible_cases(domains, feasible_case_values)

    def evaluate(self, assignment):
        full = self._augment_domains(dict(assignment))
        relevant_vars = self.variables()
        enum_items = []
        total = 1
        for name in sorted(relevant_vars):
            dom = full.get(name)
            if not isinstance(dom, list):
                continue
            lo, hi = int(dom[0]), int(dom[-1])
            width = hi - lo + 1
            if width <= 0:
                return self.default.evaluate(assignment)
            total *= width
            if total > 100000:
                _, hi_val = self.interval(assignment)
                return hi_val
            enum_items.append((name, range(lo, hi + 1)))

        if not enum_items:
            values = tuple(selector.evaluate(full) for selector in self.selectors)
            expr = self._case_map.get(values)
            if expr is not None:
                return expr.evaluate(full)
            return self.default.evaluate(full)

        best = None
        names = [name for name, _ in enum_items]
        ranges = [rng for _, rng in enum_items]
        for combo in product(*ranges):
            concrete = dict(full)
            for name, value in zip(names, combo):
                concrete[name] = value
            values = tuple(selector.evaluate(concrete) for selector in self.selectors)
            expr = self._case_map.get(values, self.default)
            val = expr.evaluate(concrete)
            best = val if best is None else max(best, val)

        if best is None:
            return self.default.evaluate(full)
        return best

    def variables(self):
        vars_in = set()
        for selector in self.selectors:
            vars_in |= selector.variables()
        for case in self.cases:
            vars_in |= case["expr"].variables()
        for lo_expr, hi_expr in self.extra_domains.values():
            vars_in |= lo_expr.variables()
            vars_in |= hi_expr.variables()
        vars_in |= self.default.variables()
        return vars_in

    def __repr__(self):
        if not self.selectors and len(self.cases) == 1 and self.cases[0]["values"] == tuple():
            return str(self.cases[0]["expr"])

        selector_texts = [str(selector) for selector in self.selectors]
        parts = []
        for case in self.cases:
            if selector_texts:
                pred = " and ".join(
                    f"({selector_text}=={target})"
                    for selector_text, target in zip(selector_texts, case["values"])
                )
            else:
                pred = "default"
            parts.append(f"[{pred}] => {case['expr']}")
        return "CaseSplit(" + "; ".join(parts) + ")"


# ─────────────────────────────────────────────────────────────
#  파서
# ─────────────────────────────────────────────────────────────
def parse_expr_tree(sym_expr_str):
    """SymExpr 문자열을 ExprNode 트리로 파싱.

    지원하는 문법:
      - 정수 리터럴
      - sp_X_Y 변수
      - a*b (곱)
      - (expr)
      - min(a,b)
      - max(a,b,...)
      - ceil(a/(b))
      - a - b, a + b (스텐실 패턴)

    Returns: ExprNode
    """
    s = sym_expr_str.strip()
    node, pos = _parse_add_sub(s, 0)
    pos = _skip_spaces(s, pos)
    if pos != len(s):
        raise ValueError(
            f"Unsupported trailing expression segment in '{sym_expr_str}': '{s[pos:]}'"
        )
    return node


def _skip_spaces(s, pos):
    while pos < len(s) and s[pos] == ' ':
        pos += 1
    return pos


def _parse_add_sub(s, pos):
    """+ / - 파싱 (가장 낮은 우선순위)."""
    left, pos = _parse_mul(s, pos)
    while True:
        pos = _skip_spaces(s, pos)
        if pos >= len(s):
            break
        if s[pos] == '+':
            right, pos = _parse_mul(s, pos + 1)
            left = AddNode(left, right)
        elif s[pos] == '-' and pos > 0:
            right, pos = _parse_mul(s, pos + 1)
            left = SubNode(left, right)
        else:
            break
    return left, pos


def _parse_mul(s, pos):
    """* 파싱."""
    left, pos = _parse_atom(s, pos)
    while True:
        pos = _skip_spaces(s, pos)
        if pos >= len(s) or s[pos] != '*':
            break
        right, pos = _parse_atom(s, pos + 1)
        left = MulNode(left, right)
    return left, pos


def _parse_atom(s, pos):
    """원자: 정수, 변수, 괄호, min(...), max(...), ceil(...)."""
    while pos < len(s) and s[pos] == ' ':
        pos += 1

    if pos >= len(s):
        return ConstNode(1), pos

    if s[pos] == '(':
        inner, pos = _parse_add_sub(s, pos + 1)
        if pos < len(s) and s[pos] == ')':
            pos += 1
        return inner, pos

    if s[pos:pos+4] == 'min(':
        pos += 4
        a, pos = _parse_add_sub(s, pos)
        if pos < len(s) and s[pos] == ',':
            pos += 1
        b, pos = _parse_add_sub(s, pos)
        if pos < len(s) and s[pos] == ')':
            pos += 1
        return MinNode(a, b), pos

    if s[pos:pos+4] == 'max(':
        pos += 4
        children = []
        while pos < len(s):
            child, pos = _parse_add_sub(s, pos)
            children.append(child)
            while pos < len(s) and s[pos] == ' ':
                pos += 1
            if pos < len(s) and s[pos] == ',':
                pos += 1
                continue
            if pos < len(s) and s[pos] == ')':
                pos += 1
            break
        return MaxNode(children), pos

    if s[pos:pos+5] == 'ceil(':
        pos += 5
        a, pos = _parse_add_sub(s, pos)
        pos = _skip_spaces(s, pos)
        if pos < len(s) and s[pos] == '/':
            pos += 1
        pos = _skip_spaces(s, pos)
        if pos < len(s) and s[pos] == '(':
            b, pos = _parse_add_sub(s, pos + 1)
            if pos < len(s) and s[pos] == ')':
                pos += 1
        else:
            b, pos = _parse_atom(s, pos)
        if pos < len(s) and s[pos] == ')':
            pos += 1
        return CeilDivNode(a, b), pos

    if s[pos:pos+10] == 'math.ceil(':
        pos += 10
        a, pos = _parse_add_sub(s, pos)
        pos = _skip_spaces(s, pos)
        if pos < len(s) and s[pos] == '/':
            pos += 1
        pos = _skip_spaces(s, pos)
        if pos < len(s) and s[pos] == '(':
            b, pos = _parse_add_sub(s, pos + 1)
            if pos < len(s) and s[pos] == ')':
                pos += 1
        else:
            b, pos = _parse_atom(s, pos)
        if pos < len(s) and s[pos] == ')':
            pos += 1
        return CeilDivNode(a, b), pos

    start = pos
    if s[pos].isdigit():
        while pos < len(s) and s[pos].isdigit():
            pos += 1
        return ConstNode(int(s[start:pos])), pos

    if s[pos].isalpha() or s[pos] == '_':
        while pos < len(s) and (s[pos].isalnum() or s[pos] == '_'):
            pos += 1
        name = s[start:pos]
        return VarNode(name), pos

    return ConstNode(1), pos + 1
