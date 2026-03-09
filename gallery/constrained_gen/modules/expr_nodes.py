"""
expr_nodes — ExprNode 트리 클래스 + parse_expr_tree 파서.

제약식 LHS를 나타내는 표현식 트리 노드.
각 노드는 partial assignment 상태에서 (lo, hi) 구간을 반환할 수 있다.
"""

builtins_min = min


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
      - ceil(a/(b))
      - a - b, a + b (스텐실 패턴)

    Returns: ExprNode
    """
    s = sym_expr_str.strip()
    return _parse_add_sub(s, 0)[0]


def _parse_add_sub(s, pos):
    """+ / - 파싱 (가장 낮은 우선순위)."""
    left, pos = _parse_mul(s, pos)
    while pos < len(s):
        if s[pos] == '+':
            right, pos = _parse_mul(s, pos + 1)
            left = AddNode(left, right)
        elif s[pos] == '-' and pos > 0:
            right, pos = _parse_mul(s, pos + 1)
            left = SubNode(left, right)
        elif s[pos] in ' ':
            j = pos
            while j < len(s) and s[j] == ' ':
                j += 1
            if j < len(s) and s[j] == '+':
                j2 = j + 1
                while j2 < len(s) and s[j2] == ' ':
                    j2 += 1
                right, pos = _parse_mul(s, j2)
                left = AddNode(left, right)
            elif j < len(s) and s[j] == '-':
                j2 = j + 1
                while j2 < len(s) and s[j2] == ' ':
                    j2 += 1
                right, pos = _parse_mul(s, j2)
                left = SubNode(left, right)
            else:
                break
        else:
            break
    return left, pos


def _parse_mul(s, pos):
    """* 파싱."""
    left, pos = _parse_atom(s, pos)
    while pos < len(s) and s[pos] == '*':
        right, pos = _parse_atom(s, pos + 1)
        left = MulNode(left, right)
    return left, pos


def _parse_atom(s, pos):
    """원자: 정수, 변수, 괄호, min(...), ceil(...)."""
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

    if s[pos:pos+5] == 'ceil(':
        pos += 5
        a, pos = _parse_add_sub(s, pos)
        if pos < len(s) and s[pos] == '/':
            pos += 1
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
        if pos < len(s) and s[pos] == '/':
            pos += 1
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
