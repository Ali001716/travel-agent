"""Agent 评估框架 — 自动运行测试用例并生成报告"""
import json
import time
import asyncio
import httpx
from datetime import datetime

CASES_FILE = "eval/cases.json"
REPORT_FILE = "eval/report.json"
API_BASE = "http://localhost:8000"


async def run_case(client: httpx.AsyncClient, case: dict) -> dict:
    """运行单个测试用例，返回评估结果"""
    tid = "eval_" + str(int(time.time() * 1000))
    body = {"message": case["message"], "thread_id": tid}
    if case.get("preferences"):
        body["preferences"] = case["preferences"]
    if case.get("location"):
        body["location"] = case["location"]

    tools_called = []
    text_output = ""
    first_token_at = None
    start_time = time.time()
    error = None

    try:
        async with client.stream(
            "POST", f"{API_BASE}/api/chat/stream",
            json=body, timeout=case.get("timeout_s", 60)
        ) as resp:
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                try:
                    data = json.loads(line[6:])
                except json.JSONDecodeError:
                    continue

                if data.get("type") == "tool_start":
                    tools_called.append(data["tool"])
                elif data.get("type") == "text":
                    if first_token_at is None:
                        first_token_at = time.time()
                    text_output += data["content"]
                elif data.get("type") == "error":
                    error = data.get("error")
    except Exception as e:
        error = str(e)

    elapsed_s = round(time.time() - start_time, 2)
    first_token_ms = round((first_token_at - start_time) * 1000) if first_token_at else None

    # 检查期望工具
    expect_tools = case.get("expect_tools", [])
    tools_hit = [t for t in expect_tools if t in tools_called]
    tool_ok = len(tools_hit) > 0

    # 检查期望关键词
    expect_keywords = case.get("expect_keywords", [])
    kw_hit = [k for k in expect_keywords if k.lower() in text_output.lower()]
    kw_ok = len(kw_hit) == len(expect_keywords) if expect_keywords else True

    return {
        "id": case["id"],
        "category": case.get("category", ""),
        "elapsed_s": elapsed_s,
        "first_token_ms": first_token_ms,
        "tools_called": tools_called,
        "tools_expected": expect_tools,
        "tools_ok": tool_ok,
        "keywords_expected": expect_keywords,
        "keywords_hit": kw_hit,
        "keywords_ok": kw_ok,
        "passed": tool_ok and kw_ok and error is None,
        "error": error,
    }


async def main():
    with open(CASES_FILE, "r", encoding="utf-8") as f:
        cases = json.load(f)

    print(f"运行 {len(cases)} 个测试用例...\n")

    results = []
    for i, case in enumerate(cases, 1):
        async with httpx.AsyncClient() as client:
            print(f"[{i}/{len(cases)}] {case['id']} ... ", end="", flush=True)
            result = await run_case(client, case)
            status = "PASS" if result["passed"] else "FAIL"
            print(f"{status} ({result['elapsed_s']}s, "
                  f"tools={result['tools_called']}, "
                  f"first_token={result['first_token_ms']}ms)")
            if result["error"]:
                print(f"       ERROR: {result['error'][:100]}")
            results.append(result)

    # 统计
    passed = sum(1 for r in results if r["passed"])
    total = len(results)
    tool_accuracy = sum(1 for r in results if r["tools_ok"]) / total * 100
    kw_accuracy = sum(1 for r in results if r["keywords_ok"]) / total * 100
    first_tokens = [r["first_token_ms"] for r in results if r["first_token_ms"]]
    elapsed_times = [r["elapsed_s"] for r in results]

    report = {
        "meta": {
            "timestamp": datetime.now().isoformat(),
            "total_cases": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate": f"{passed / total * 100:.1f}%",
        },
        "metrics": {
            "tool_accuracy": f"{tool_accuracy:.1f}%",
            "keyword_accuracy": f"{kw_accuracy:.1f}%",
            "first_token_avg_ms": round(sum(first_tokens) / len(first_tokens)) if first_tokens else None,
            "first_token_min_ms": min(first_tokens) if first_tokens else None,
            "first_token_max_ms": max(first_tokens) if first_tokens else None,
            "total_elapsed_avg_s": round(sum(elapsed_times) / len(elapsed_times), 1),
            "total_elapsed_min_s": min(elapsed_times),
            "total_elapsed_max_s": max(elapsed_times),
            "error_rate": f"{sum(1 for r in results if r['error']) / total * 100:.1f}%",
        },
        "by_category": {},
        "cases": results,
    }

    # 按类别统计
    categories = {}
    for r in results:
        cat = r.get("category", "未分类")
        if cat not in categories:
            categories[cat] = {"total": 0, "passed": 0, "avg_elapsed_s": 0}
        categories[cat]["total"] += 1
        if r["passed"]:
            categories[cat]["passed"] += 1
        categories[cat]["avg_elapsed_s"] = round(
            (categories[cat]["avg_elapsed_s"] * (categories[cat]["total"] - 1) + r["elapsed_s"])
            / categories[cat]["total"], 1
        )
    report["by_category"] = categories

    # 保存报告
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # 打印摘要
    print(f"\n{'='*50}")
    print(f"评估完成")
    print(f"通过率: {report['meta']['pass_rate']} ({passed}/{total})")
    print(f"工具准确率: {report['metrics']['tool_accuracy']}")
    print(f"关键词匹配率: {report['metrics']['keyword_accuracy']}")
    print(f"首Token: 平均{report['metrics']['first_token_avg_ms']}ms "
          f"(最快{report['metrics']['first_token_min_ms']}ms / "
          f"最慢{report['metrics']['first_token_max_ms']}ms)")
    print(f"总耗时: 平均{report['metrics']['total_elapsed_avg_s']}s "
          f"(最快{report['metrics']['total_elapsed_min_s']}s / "
          f"最慢{report['metrics']['total_elapsed_max_s']}s)")
    print(f"错误率: {report['metrics']['error_rate']}")
    print(f"\n报告已保存: {REPORT_FILE}")


if __name__ == "__main__":
    asyncio.run(main())
