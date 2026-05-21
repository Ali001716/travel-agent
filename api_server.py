import json
import logging
import os
import sqlite3
import time
from contextlib import asynccontextmanager

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from langchain_core.messages import HumanMessage
from langchain_deepseek import ChatDeepSeek
from langchain_mcp_adapters.client import MultiServerMCPClient
import aiosqlite

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel

from observability import TraceContext, get_stats, get_recent_requests

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_servers_config():
    with open("servers_config.json", "r") as f:
        return json.load(f)


def load_prompt():
    with open("agent_prompts.txt", "r", encoding="utf-8") as f:
        return f.read()


mcp_client = None
agent = None
db_conn = None
session_db = None  # SQLite 连接，存储会话元数据
search_cache = {}
prompt = load_prompt()


class LocationInfo(BaseModel):
    lat: float
    lng: float
    address: str = ""


class ChatRequest(BaseModel):
    message: str
    thread_id: str = "default_session"
    location: LocationInfo | None = None
    preferences: dict | None = None


class ChatResponse(BaseModel):
    response: str
    error: str | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global mcp_client, agent, db_conn, session_db

    logger.info("正在启动 MCP Agent 服务...")

    # 初始化会话元数据库
    session_db = sqlite3.connect("sessions.db", check_same_thread=False)
    session_db.execute(
        "CREATE TABLE IF NOT EXISTS sessions ("
        "  thread_id TEXT PRIMARY KEY,"
        "  title TEXT,"
        "  created_at TEXT,"
        "  updated_at TEXT"
        ")"
    )
    session_db.commit()

    try:
        servers_cfg = load_servers_config()
        mcp_client = MultiServerMCPClient(servers_cfg)
        tools = await mcp_client.get_tools()
        logger.info("已加载 %d 个工具: %s", len(tools), [t.name for t in tools])

        model = ChatDeepSeek(
            model="deepseek-chat",
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            api_base=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
        )

        db_conn = await aiosqlite.connect("checkpoints.db")
        checkpointer = AsyncSqliteSaver(db_conn)
        agent = create_react_agent(
            model=model,
            tools=tools,
            prompt=prompt,
            checkpointer=checkpointer,
        )
        logger.info("Agent 初始化完成")
    except Exception as e:
        logger.error("服务启动失败: %s", e)
        raise

    yield

    logger.info("正在清理资源...")
    if session_db:
        session_db.close()
    if db_conn:
        await db_conn.close()
    if mcp_client:
        try:
            await mcp_client.cleanup()
        except Exception as e:
            logger.warning("清理 MCP 客户端时出错: %s", e)


app = FastAPI(lifespan=lifespan, title="MCP Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse)
async def root():
    with open("static/index.html", "r", encoding="utf-8") as f:
        content = f.read()
    return HTMLResponse(
        content=content,
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@app.get("/api/search-results/{search_id}")
async def get_search_results(search_id: str):
    """获取缓存的搜索结果（含坐标）"""
    try:
        with open("static/search_cache.json", "r", encoding="utf-8") as f:
            cache = json.load(f)
        return cache.get(search_id, [])
    except Exception:
        return []


@app.get("/api/messages/{thread_id}")
async def get_messages(thread_id: str):
    """读取会话历史消息"""
    try:
        async with AsyncSqliteSaver.from_conn_string("checkpoints.db") as saver:
            state = await saver.aget_tuple({"configurable": {"thread_id": thread_id}})
            if not state or not state.checkpoint:
                return []
            msgs = state.checkpoint.get("channel_values", {}).get("messages", [])
            result = []
            for m in msgs:
                role = getattr(m, "type", "unknown")
                content = getattr(m, "content", "")
                if role == "tool":
                    continue  # 跳过工具消息
                result.append({"role": role, "content": str(content)})
            return result
    except Exception as e:
        logger.warning("读取消息失败: %s", e)
        return []


@app.post("/api/sessions")
async def create_session():
    """创建新会话并返回 thread_id"""
    tid = "session_" + str(int(time.time() * 1000))
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    session_db.execute(
        "INSERT OR REPLACE INTO sessions(thread_id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
        (tid, "新对话", now, now),
    )
    session_db.commit()
    return {"thread_id": tid}


@app.get("/api/sessions")
async def list_sessions():
    """会话列表（按最近活跃排序）"""
    rows = session_db.execute(
        "SELECT thread_id, title, created_at, updated_at FROM sessions ORDER BY updated_at DESC LIMIT 50"
    ).fetchall()
    return [
        {"thread_id": r[0], "title": r[1], "created_at": r[2], "updated_at": r[3]}
        for r in rows
    ]


class RenameRequest(BaseModel):
    title: str


@app.put("/api/sessions/{thread_id}")
async def rename_session(thread_id: str, req: RenameRequest):
    """重命名会话"""
    session_db.execute(
        "UPDATE sessions SET title = ? WHERE thread_id = ?",
        (req.title[:100], thread_id),
    )
    session_db.commit()
    return {"ok": True}


@app.delete("/api/sessions/{thread_id}")
async def delete_session(thread_id: str):
    """删除会话及其对话记录"""
    session_db.execute("DELETE FROM sessions WHERE thread_id = ?", (thread_id,))
    session_db.commit()
    # 清掉 LangGraph checkpoint
    try:
        await db_conn.execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,))
        await db_conn.execute("DELETE FROM writes WHERE thread_id = ?", (thread_id,))
        await db_conn.commit()
    except Exception as e:
        logger.warning("删除 checkpoint 失败: %s", e)
    return {"ok": True}


@app.get("/api/kb/stats")
async def kb_stats_endpoint():
    """知识库统计"""
    from knowledge_base import get_stats as kb_stats
    return kb_stats()


@app.get("/api/stats")
async def stats(thread_id: str = None):
    """Agent 可观测性统计"""
    return get_stats(thread_id)


@app.get("/api/stats/recent")
async def recent(n: int = 50):
    """最近 N 条请求详情"""
    return get_recent_requests(n)


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    """可观测性监控面板"""
    return HTMLResponse("""
<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Agent 监控面板</title>
<style>
  :root { --bg: #0f172a; --card: #1e293b; --text: #e2e8f0; --muted: #94a3b8;
          --green: #22c55e; --red: #ef4444; --blue: #3b82f6; --border: #334155; }
  * { margin:0; padding:0; box-sizing:border-box; }
  body { font-family: -apple-system, sans-serif; background: var(--bg); color: var(--text); padding: 20px; }
  h1 { font-size: 20px; margin-bottom: 16px; }
  .cards { display: flex; gap: 12px; margin-bottom: 20px; flex-wrap: wrap; }
  .card { background: var(--card); border-radius: 10px; padding: 16px 20px; min-width: 140px; border: 1px solid var(--border); }
  .card .val { font-size: 28px; font-weight: 700; margin-top: 4px; }
  .card .lbl { font-size: 12px; color: var(--muted); }
  .green { color: var(--green); } .red { color: var(--red); } .blue { color: var(--blue); }

  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th { text-align: left; padding: 8px 12px; border-bottom: 1px solid var(--border); color: var(--muted); font-weight: 500; }
  td { padding: 8px 12px; border-bottom: 1px solid var(--border); }
  tr:hover { background: #1e293b; }
  .tool-tag { display: inline-block; padding: 1px 6px; margin: 1px 2px; border-radius: 4px; font-size: 11px; background: #1e3a5f; color: #60a5fa; }
  .err-tag { background: #7f1d1d; color: #fca5a5; }
  .refresh { font-size: 12px; color: var(--muted); margin-left: 10px; }
</style>
</head>
<body>
<h1>🔍 Agent 可观测性 <span class="refresh">（每 3 秒自动刷新）</span></h1>

<div class="cards" id="summary"></div>
<table>
  <thead>
    <tr><th>时间</th><th>会话</th><th>消息</th><th>耗时</th><th>Token</th><th>LLM</th><th>工具调用</th></tr>
  </thead>
  <tbody id="rows"></tbody>
</table>

<script>
async function load() {
  const [statsResp, recentResp] = await Promise.all([
    fetch('/api/stats'), fetch('/api/stats/recent?n=30')
  ]);
  const stats = await statsResp.json();
  const data = await recentResp.json();

  // Summary cards
  const summary = document.getElementById('summary');
  const ver = data.has_langfuse ? 'LangFuse 云' : '本地';
  summary.innerHTML =
    '<div class="card"><div class="lbl">请求总数</div><div class="val blue">' + (stats.total_requests||0) + '</div></div>' +
    '<div class="card"><div class="lbl">活跃会话</div><div class="val">' + (stats.total_sessions||0) + '</div></div>' +
    '<div class="card"><div class="lbl">平均耗时</div><div class="val">' + (stats.avg_elapsed_s||0) + 's</div></div>' +
    '<div class="card"><div class="lbl">Token 消耗</div><div class="val">' + (stats.total_tokens||0) + '</div></div>' +
    '<div class="card"><div class="lbl">工具调用</div><div class="val">' + (stats.total_tool_calls||0) + '</div></div>' +
    '<div class="card"><div class="lbl">追踪模式</div><div class="val" style="font-size:16px">' + ver + '</div></div>';

  // Request rows
  const rows = document.getElementById('rows');
  rows.innerHTML = (data.requests||[]).map(r => {
    let toolsHtml = (r.tool_calls||[]).map(t =>
      '<span class="tool-tag' + (t.success ? '' : ' err-tag') + '">' + t.tool + ' ' + t.duration_ms + 'ms</span>'
    ).join('');
    const errStyle = r.error ? ' style="color:#fca5a5"' : '';
    return '<tr' + errStyle + '>' +
      '<td>' + r.time + '</td>' +
      '<td>' + r.thread_id.substring(0, 12) + '</td>' +
      '<td>' + r.message.substring(0, 60) + '</td>' +
      '<td>' + r.elapsed_s + 's</td>' +
      '<td>' + (r.tokens||0) + '</td>' +
      '<td>' + r.llm_calls + '次</td>' +
      '<td>' + toolsHtml + '</td>' +
      '</tr>';
  }).join('');
}
load();
setInterval(load, 3000);
</script>
</body>
</html>""")


@app.get("/api/health")
async def health():
    return {
        "message": "MCP Agent API 运行中",
        "status": "healthy" if agent else "degraded",
        "agent_ready": agent is not None,
    }


PREF_LABELS = {
    "scene": {"户外自然": "喜欢户外自然", "室内文化": "喜欢室内文化场馆"},
    "pace": {"悠闲慢游": "节奏悠闲，每天2-3个点", "紧凑打卡": "紧凑充实，每天4-5个点"},
    "food": {"地道小吃": "偏好地道小吃", "高端餐厅": "偏好高端餐厅", "素食": "素食者"},
    "companion": {"独自": "独自旅行", "情侣": "情侣出游", "亲子带娃": "亲子游，带孩子", "朋友结伴": "朋友结伴出游"},
    "budget": {"穷游省钱": "预算敏感，省钱为主", "舒适中等": "中等舒适预算", "奢华享受": "预算充裕，追求品质"},
    "transport": {"地铁公交优先": "优先地铁公交", "打车方便": "偏好打车", "步行友好": "偏好步行"},
    "web": {"开启": "允许联网搜索", "关闭": "禁止联网搜索，只用本地知识库"},
}


def build_message(message: str, location: LocationInfo | None, preferences: dict | None = None) -> str:
    """注入位置上下文和偏好设置"""
    parts = []

    if location and location.lat and location.lng:
        parts.append(
            f"[用户当前位置: {location.address or f'{location.lat},{location.lng}'} "
            f"(坐标: {location.lng},{location.lat})]"
        )

    if preferences:
        pref_texts = []
        for key, value in preferences.items():
            if key in PREF_LABELS and value in PREF_LABELS[key]:
                pref_texts.append(PREF_LABELS[key][value])
        if pref_texts:
            parts.append("[用户偏好: " + "，".join(pref_texts) + "]")

    if parts:
        return "\n".join(parts) + "\n\n" + message
    return message


@app.get("/api/location")
async def reverse_geocode(lat: float, lng: float):
    """反向地理编码：经纬度 → 地址"""
    amap_key = os.getenv("AMAP_API_KEY", "")
    if not amap_key:
        return {"error": "未配置 AMAP_API_KEY", "address": f"{lat},{lng}"}

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                "https://restapi.amap.com/v3/geocode/regeo",
                params={"key": amap_key, "location": f"{lng},{lat}"},
            )
            data = resp.json()
        if data.get("status") == "1" and data.get("regeocode"):
            addr = data["regeocode"].get("formatted_address", "")
            return {"address": addr}
    except Exception as e:
        logger.warning("反向地理编码失败: %s", e)

    return {"address": f"{lat},{lng}"}


@app.get("/api/location/manual")
async def geocode_address(address: str):
    """手动地址解析：地名/地址 → 经纬度（POI 搜索 + 地理编码双保险）"""
    amap_key = os.getenv("AMAP_API_KEY", "")
    if not amap_key:
        return {"error": "未配置 AMAP_API_KEY"}

    try:
        async with httpx.AsyncClient() as client:
            # 先 POI 搜索（适合地名）
            poi_resp = await client.get(
                "https://restapi.amap.com/v3/place/text",
                params={"key": amap_key, "keywords": address},
            )
            poi_data = poi_resp.json()
            if poi_data.get("status") == "1" and poi_data.get("pois"):
                poi = poi_data["pois"][0]
                loc = poi["location"].split(",")
                return {"lat": float(loc[1]), "lng": float(loc[0]), "address": poi.get("name", address)}

            # 回退：地理编码
            geo_resp = await client.get(
                "https://restapi.amap.com/v3/geocode/geo",
                params={"key": amap_key, "address": address},
            )
            geo_data = geo_resp.json()
            if geo_data.get("status") == "1" and geo_data.get("geocodes"):
                geo = geo_data["geocodes"][0]
                loc = geo["location"].split(",")
                return {"lat": float(loc[1]), "lng": float(loc[0]), "address": geo.get("formatted_address", address)}

            return {"error": f"找不到该地址: {address}"}
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    if not agent:
        return ChatResponse(response="", error="Agent 未就绪，请稍后重试")

    user_msg = build_message(request.message, request.location, request.preferences)
    logger.info("[%s] 收到消息: %s", request.thread_id, request.message[:100])

    start = time.time()
    try:
        result = await agent.ainvoke(
            {"messages": [HumanMessage(content=user_msg)]},
            {"configurable": {"thread_id": request.thread_id}, "recursion_limit": 100},
        )
        reply = result["messages"][-1].content
        elapsed = time.time() - start
        logger.info("[%s] 回复完成 (%.2fs)", request.thread_id, elapsed)
        return ChatResponse(response=reply)
    except Exception as e:
        elapsed = time.time() - start
        logger.error("[%s] 请求失败 (%.2fs): %s", request.thread_id, elapsed, e)
        return ChatResponse(response="", error=f"处理请求时出错: {str(e)}")


@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    """流式聊天接口 (SSE)"""
    if not agent:
        async def error_stream():
            yield f"data: {json.dumps({'error': 'Agent 未就绪'})}\n\n"
        return StreamingResponse(error_stream(), media_type="text/event-stream")

    async def event_stream():
        user_msg = build_message(request.message, request.location, request.preferences)
        # 保存会话元数据
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        session_db.execute(
            "INSERT OR IGNORE INTO sessions(thread_id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (request.thread_id, request.message[:80], now, now),
        )
        session_db.execute(
            "UPDATE sessions SET updated_at = ? WHERE thread_id = ?",
            (now, request.thread_id),
        )
        session_db.commit()

        trace = TraceContext(request.thread_id, user_msg)
        tool_timers: dict[str, float] = {}
        logger.info("[%s] 收到消息(stream): %s", request.thread_id, request.message[:100])
        start = time.time()

        try:
            async for event in agent.astream_events(
                {"messages": [HumanMessage(content=user_msg)]},
                {"configurable": {"thread_id": request.thread_id}, "recursion_limit": 100},
                version="v2",
            ):
                kind = event["event"]

                # LLM 流式 token
                if kind == "on_chat_model_stream":
                    chunk = event["data"]["chunk"]
                    if hasattr(chunk, "content") and chunk.content:
                        trace.record_llm(1)
                        yield f"data: {json.dumps({'type': 'text', 'content': chunk.content})}\n\n"

                # 工具调用开始
                elif kind == "on_tool_start":
                    tool_name = event.get("name", "unknown")
                    tool_timers[tool_name] = time.time()
                    yield f"data: {json.dumps({'type': 'tool_start', 'tool': tool_name})}\n\n"

                # 工具调用结束
                elif kind == "on_tool_end":
                    tool_name = event.get("name", "unknown")
                    duration = (time.time() - tool_timers.pop(tool_name, time.time())) * 1000
                    trace.record_tool(tool_name, duration, True)
                    tool_output = str(event["data"].get("output", ""))
                    preview = tool_output[:200]
                    yield f"data: {json.dumps({'type': 'tool_end', 'tool': tool_name, 'preview': preview})}\n\n"

                    if tool_name in ("search_places", "search_nearby"):
                        import re
                        m = re.search(r"\{search_id:(\w+)\}", tool_output)
                        if m:
                            yield f"data: {json.dumps({'type': 'search_ready', 'search_id': m.group(1)})}\n\n"

            trace.finish()
            elapsed = time.time() - start
            logger.info("[%s] 流式回复完成 (%.2fs) | LLM:%d次 Tool:%d个 Token:%d",
                        request.thread_id, elapsed, trace.llm_calls, len(trace.tool_calls), trace.total_tokens)
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        except Exception as e:
            trace.set_error(str(e))
            trace.finish()
            elapsed = time.time() - start
            logger.error("[%s] 流式请求失败 (%.2fs): %s", request.thread_id, elapsed, e)
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn

    logger.info("打开浏览器访问 http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
