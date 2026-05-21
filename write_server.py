from mcp.server.fastmcp import FastMCP
import os
from datetime import datetime

mcp = FastMCP("WriteServer")


@mcp.tool()
async def write_file(filename: str, content: str) -> str:
    """
    将内容写入到本地文件。
    :param filename: 文件名（会自动保存到 output 文件夹）
    :param content: 要写入的内容
    :return: 操作结果
    """
    try:
        # 创建输出目录
        os.makedirs("output", exist_ok=True)

        # 添加时间戳避免覆盖
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name, ext = os.path.splitext(filename)
        final_filename = f"{name}_{timestamp}{ext}"
        filepath = os.path.join("output", final_filename)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

        return f"✅ 文件已保存: {filepath}"
    except Exception as e:
        return f"❌ 保存失败: {str(e)}"


@mcp.tool()
async def read_file(filename: str) -> str:
    """
    读取本地文件内容。
    :param filename: 文件名
    :return: 文件内容
    """
    try:
        filepath = os.path.join("output", filename)
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        return f"文件内容:\n{content}"
    except FileNotFoundError:
        return f"❌ 文件不存在: {filename}"
    except Exception as e:
        return f"❌ 读取失败: {str(e)}"


if __name__ == "__main__":
    mcp.run(transport='stdio')