import asyncio
from langgraph.prebuilt import create_react_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_deepseek import ChatDeepSeek
import sqlite3

from langgraph.checkpoint.sqlite import SqliteSaver
import json
import os
from dotenv import load_dotenv

load_dotenv()


def load_servers_config():
    with open("servers_config.json", "r") as f:
        return json.load(f)


def load_prompt():
    with open("agent_prompts.txt", "r", encoding="utf-8") as f:
        return f.read()


async def run_chat_loop():
    """启动 MCP-Agent 聊天循环"""
    print("🤖 MCP Agent 交互式聊天客户端")
    print("=" * 50)
    print("输入 'quit' 退出程序")
    print("输入 'clear' 清除对话历史")
    print("-" * 50)

    # 1. 连接多台 MCP 服务器
    servers_cfg = load_servers_config()
    prompt = load_prompt()

    print("📡 正在连接 MCP 服务器...")
    mcp_client = MultiServerMCPClient(servers_cfg)
    tools = await mcp_client.get_tools()
    print(f"✅ 已连接，加载了 {len(tools)} 个工具:")
    for tool in tools:
        print(f"   - {tool.name}")

    # 2. 初始化大模型
    model = ChatDeepSeek(
        model="deepseek-chat",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        api_base=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
    )

    # 3. 构造 Agent（带记忆）
    checkpointer = SqliteSaver(
        sqlite3.connect("checkpoints.db", check_same_thread=False)
    )
    agent = create_react_agent(
        model=model,
        tools=tools,
        prompt=prompt,
        checkpointer=checkpointer
    )
    print("\n✨ Agent 已就绪，开始对话吧！\n")

    # 4. CLI 聊天循环
    thread_id = "1"

    while True:
        user_input = input("\n👤 你: ").strip()
        if user_input.lower() == "quit":
            print("👋 再见！")
            break
        if user_input.lower() == "clear":
            thread_id = str(int(thread_id) + 1)
            print("🗑️ 对话历史已清除")
            continue
        if not user_input:
            continue

        print("🤔 思考中...")
        try:
            result = await agent.ainvoke(
                {"messages": [{"role": "user", "content": user_input}]},
                config={"configurable": {"thread_id": thread_id}, "recursion_limit": 100}
            )
            print(f"\n🤖 AI: {result['messages'][-1].content}")
        except Exception as e:
            print(f"❌ 错误: {e}")

    # 5. 清理
    await mcp_client.cleanup()


if __name__ == "__main__":
    asyncio.run(run_chat_loop())