import ast
import math
import operator
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("CalculatorServer")

# 允许的运算符和函数白名单
SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}

SAFE_FUNCS = {
    "sqrt": math.sqrt,
    "pow": math.pow,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "abs": abs,
    "round": round,
}

SAFE_CONST = {
    "pi": math.pi,
    "e": math.e,
}


def safe_eval(expr: str) -> float:
    """使用 AST 安全计算数学表达式"""
    tree = ast.parse(expr.strip(), mode="eval")

    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        elif isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError(f"不支持的常量: {node.value}")
        elif isinstance(node, ast.BinOp):
            left = _eval(node.left)
            right = _eval(node.right)
            op_type = type(node.op)
            if op_type not in SAFE_OPS:
                raise ValueError(f"不支持的运算符: {op_type.__name__}")
            return SAFE_OPS[op_type](left, right)
        elif isinstance(node, ast.UnaryOp):
            operand = _eval(node.operand)
            op_type = type(node.op)
            if op_type not in SAFE_OPS:
                raise ValueError(f"不支持的运算符: {op_type.__name__}")
            return SAFE_OPS[op_type](operand)
        elif isinstance(node, ast.Call):
            func_name = node.func.id if isinstance(node.func, ast.Name) else None
            if func_name not in SAFE_FUNCS:
                raise ValueError(f"不支持的函数: {func_name}")
            args = [_eval(a) for a in node.args]
            return SAFE_FUNCS[func_name](*args)
        elif isinstance(node, ast.Name):
            if node.id in SAFE_CONST:
                return SAFE_CONST[node.id]
            raise ValueError(f"不支持的变量: {node.id}")
        else:
            raise ValueError(f"不支持的语法: {type(node).__name__}")

    return _eval(tree)


@mcp.tool()
async def calculate(expression: str) -> str:
    """
    计算数学表达式。
    :param expression: 数学表达式，如 "2 + 3 * 4" 或 "sqrt(16) + pow(2,3)"
    :return: 计算结果
    """
    try:
        result = safe_eval(expression)
        return f"计算结果: {expression} = {result}"
    except Exception as e:
        return f"计算错误: {str(e)}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
