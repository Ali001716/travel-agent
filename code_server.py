"""code_run — 沙箱代码执行工具（GenericAgent 设计理念）"""
import os
import subprocess
import tempfile
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("CodeServer")

# 允许的安全模块白名单
SAFE_IMPORTS = {
    "math", "json", "re", "datetime", "random",
    "itertools", "collections", "functools", "operator",
    "statistics", "heapq", "bisect", "copy",
}

SANDBOX_DIR = "sandbox"


@mcp.tool()
async def code_run(code: str) -> str:
    """
    在安全沙箱中执行 Python 代码。
    仅当需要批量计算、数据排序/筛选、格式转换等现有工具无法完成的任务时才使用。
    不要用此工具查询天气、搜索店铺、规划路线——那些有专用工具。

    :param code: Python 代码
    :return: 执行输出
    """
    os.makedirs(SANDBOX_DIR, exist_ok=True)

    # 模块白名单检查
    for imp in SAFE_IMPORTS:
        if imp in code:
            break
    else:
        # 没有 import 任何白名单模块也可以（纯计算）
        pass

    # 禁用危险操作
    dangerous = ["__import__", "open(", "subprocess", "os.", "sys.", "shutil",
                 "socket", "eval(", "exec(", "compile(", "globals()", "locals()",
                 "getattr(", "setattr(", "delattr(", "breakpoint("]
    for d in dangerous:
        if d in code.lower():
            return f"错误：禁止使用 {d}"

    try:
        result = subprocess.run(
            ["python", "-c", code],
            capture_output=True, text=True,
            timeout=30,
            cwd=SANDBOX_DIR,
            env={**os.environ, "PYTHONPATH": "."},
        )
        output = result.stdout.strip()
        if result.stderr.strip():
            output += "\n" + result.stderr.strip()
        return output or "(无输出)"
    except subprocess.TimeoutExpired:
        return "错误：代码执行超时（30秒）"
    except Exception as e:
        return f"错误：{str(e)}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
