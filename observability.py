"""Agent 可观测性 — 基于 LangFuse 的追踪与本地监控"""
import os
import time
import json
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

_langfuse = None
_sessions = defaultdict(list)  # thread_id → list of request records
_requests = []  # 全局请求列表（最近 200 条）

try:
    pk = os.getenv("LANGFUSE_PUBLIC_KEY")
    sk = os.getenv("LANGFUSE_SECRET_KEY")
    host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
    if pk and sk:
        from langfuse import Langfuse
        _langfuse = Langfuse(public_key=pk, secret_key=sk, host=host)
        logger.info("LangFuse 已连接: %s", host)
    else:
        logger.info("未配置 LangFuse，使用本地监控模式")
except Exception as e:
    logger.warning("LangFuse 初始化失败: %s", e)


class TraceContext:
    """单次请求的追踪上下文"""

    def __init__(self, thread_id: str, message: str):
        self.thread_id = thread_id
        self.start_time = time.time()
        self.llm_calls = 0
        self.tool_calls: list[dict] = []
        self.total_tokens = 0
        self.message = message[:200]
        self.error = None

        self._trace = None
        if _langfuse:
            try:
                self._trace = _langfuse.trace(
                    name="chat",
                    session_id=thread_id,
                    input={"message": self.message},
                )
            except Exception as e:
                logger.debug("LangFuse trace 创建失败: %s", e)

    def record_llm(self, tokens: int):
        self.llm_calls += 1
        self.total_tokens += tokens

    def record_tool(self, tool_name: str, duration_ms: float, success: bool):
        self.tool_calls.append({
            "tool": tool_name,
            "duration_ms": round(duration_ms, 1),
            "success": success,
        })

    def set_error(self, error: str):
        self.error = error

    def finish(self):
        elapsed = round(time.time() - self.start_time, 2)

        if self._trace:
            try:
                self._trace.update(
                    output={"elapsed_s": elapsed, "error": self.error},
                    metadata={
                        "llm_calls": self.llm_calls,
                        "tool_calls": len(self.tool_calls),
                        "total_tokens": self.total_tokens,
                        "tools": [t["tool"] for t in self.tool_calls],
                    },
                )
            except Exception:
                pass

        record = {
            "time": time.strftime("%H:%M:%S"),
            "thread_id": self.thread_id,
            "message": self.message,
            "elapsed_s": elapsed,
            "llm_calls": self.llm_calls,
            "tool_calls": self.tool_calls,
            "tool_count": len(self.tool_calls),
            "tokens": self.total_tokens,
            "error": self.error,
        }

        _sessions[self.thread_id].append(record)
        _requests.insert(0, record)
        if len(_requests) > 200:
            _requests.pop()


def get_stats(thread_id: str = None):
    """全局 / 会话统计"""
    if thread_id and thread_id in _sessions:
        records = _sessions[thread_id]
        return {
            "thread_id": thread_id,
            "total_requests": len(records),
            "avg_elapsed_s": round(sum(r["elapsed_s"] for r in records) / len(records), 2),
            "total_tokens": sum(r["tokens"] for r in records),
            "total_tool_calls": sum(r["tool_count"] for r in records),
            "error_rate": f"{sum(1 for r in records if r['error']) / len(records) * 100:.1f}%",
        }

    all_records = _requests
    if not all_records:
        return {"status": "no data"}
    return {
        "total_sessions": len(_sessions),
        "total_requests": len(all_records),
        "avg_elapsed_s": round(sum(r["elapsed_s"] for r in all_records) / len(all_records), 2),
        "total_tokens": sum(r["tokens"] for r in all_records),
        "total_tool_calls": sum(r["tool_count"] for r in all_records),
        "has_langfuse": _langfuse is not None,
    }


def get_recent_requests(n: int = 50):
    """获取最近 N 条请求详情"""
    return {
        "requests": _requests[:n],
        "total_sessions": len(_sessions),
        "has_langfuse": _langfuse is not None,
    }
