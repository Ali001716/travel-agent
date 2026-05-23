"""Agent 评估框架 — 自动运行测试用例 + 对比历史基线"""
import json
import time
import os
import sys
import asyncio
import httpx
from datetime import datetime

CASES_FILE = "eval/cases.json"
REPORT_FILE = "eval/report.json"
HISTORY_DIR = "eval/history"
BASELINE_FILE = "eval/baseline.json"
API_BASE = "http://localhost:8000"


async def run_case(client: httpx.AsyncClient, case: dict) -> dict:
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

    expect_tools = case.get("expect_tools", [])
    tools_hit = [t for t in expect_tools if t in tools_called]
    tool_ok = len(tools_hit) > 0

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


def build_report(results, label=""):
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    first_tokens = [r["first_token_ms"] for r in results if r["first_token_ms"]]
    elapsed_times = [r["elapsed_s"] for r in results]

    return {
        "meta": {
            "timestamp": datetime.now().isoformat(),
            "label": label,
            "total_cases": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate": f"{passed / total * 100:.1f}%",
        },
        "metrics": {
            "tool_accuracy": f"{sum(1 for r in results if r['tools_ok']) / total * 100:.1f}%",
            "keyword_accuracy": f"{sum(1 for r in results if r['keywords_ok']) / total * 100:.1f}%",
            "first_token_avg_ms": round(sum(first_tokens) / len(first_tokens)) if first_tokens else None,
            "first_token_min_ms": min(first_tokens) if first_tokens else None,
            "first_token_max_ms": max(first_tokens) if first_tokens else None,
            "total_elapsed_avg_s": round(sum(elapsed_times) / len(elapsed_times), 1),
            "total_elapsed_min_s": min(elapsed_times) if elapsed_times else None,
            "total_elapsed_max_s": max(elapsed_times) if elapsed_times else None,
            "error_rate": f"{sum(1 for r in results if r['error']) / total * 100:.1f}%",
        },
        "by_category": {},
        "cases": results,
    }


def calc_category(report):
    cats = {}
    for r in report["cases"]:
        cat = r.get("category", "未分类")
        if cat not in cats:
            cats[cat] = {"total": 0, "passed": 0, "avg_elapsed_s": 0}
        cats[cat]["total"] += 1
        if r["passed"]:
            cats[cat]["passed"] += 1
        cats[cat]["avg_elapsed_s"] = round(
            (cats[cat]["avg_elapsed_s"] * (cats[cat]["total"] - 1) + r["elapsed_s"])
            / cats[cat]["total"], 1
        )
    return cats


def print_report(report: dict):
    m = report["metrics"]
    n = report["meta"]
    print(f"\n{'='*55}")
    print(f"  {n.get('label', '评估结果')}")
    print(f"{'='*55}")
    print(f"  通过率:   {n['pass_rate']} ({n['passed']}/{n['total_cases']})")
    print(f"  工具准确率: {m['tool_accuracy']}")
    print(f"  关键词率:   {m['keyword_accuracy']}")
    print(f"  首Token:   avg {m['first_token_avg_ms']}ms  "
          f"(min {m['first_token_min_ms']}ms / max {m['first_token_max_ms']}ms)")
    print(f"  总耗时:    avg {m['total_elapsed_avg_s']}s  "
          f"(min {m['total_elapsed_min_s']}s / max {m['total_elapsed_max_s']}s)")
    print(f"  错误率:   {m['error_rate']}")


def print_diff(new_report: dict, old_report: dict):
    """打印与基线的对比"""
    nm, om = new_report["metrics"], old_report["metrics"]
    nn, on = new_report["meta"], old_report["meta"]

    def delta_str(new_val, old_val, suffix="", lower_is_better=False):
        try:
            nv = float(str(new_val).rstrip("%"))
            ov = float(str(old_val).rstrip("%"))
            diff = nv - ov
            arrow = "↓" if (lower_is_better and diff < 0) or (not lower_is_better and diff > 0) else "↑" if diff != 0 else "→"
            return f"{new_val} ({arrow}{abs(diff):.1f}{suffix})"
        except (ValueError, AttributeError):
            return f"{new_val} (was {old_val})"

    print(f"\n{'='*55}")
    print(f"  优化前后对比")
    print(f"{'='*55}")
    print(f"  {'指标':<20} {'优化前':<20} {'优化后':<20}")
    print(f"  {'-'*20} {'-'*20} {'-'*20}")
    print(f"  {'通过率':<20} {on['pass_rate']:<20} {nn['pass_rate']:<20}")
    print(f"  {'工具准确率':<20} {om['tool_accuracy']:<20} {nm['tool_accuracy']:<20}")
    print(f"  {'关键词率':<20} {om['keyword_accuracy']:<20} {nm['keyword_accuracy']:<20}")
    print(f"  {'首Token(avg)':<20} {str(om['first_token_avg_ms'])+'ms':<20} {str(nm['first_token_avg_ms'])+'ms':<20}")
    print(f"  {'总耗时(avg)':<20} {str(om['total_elapsed_avg_s'])+'s':<20} {str(nm['total_elapsed_avg_s'])+'s':<20}")
    print(f"  {'错误率':<20} {om['error_rate']:<20} {nm['error_rate']:<20}")


async def main():
    # 加载用例
    with open(CASES_FILE, "r", encoding="utf-8") as f:
        cases = json.load(f)

    label = sys.argv[1] if len(sys.argv) > 1 else datetime.now().strftime("%m%d_%H%M")

    print(f"\n  Agent 评估 — {label}")
    print(f"  用例数: {len(cases)}")

    # 运行
    results = []
    for i, case in enumerate(cases, 1):
        async with httpx.AsyncClient(trust_env=False) as client:
            print(f"  [{i}/{len(cases)}] {case['id']:<25} ", end="", flush=True)
            result = await run_case(client, case)
            status = "PASS" if result["passed"] else "FAIL"
            print(f"{status}  {result['elapsed_s']}s  token@{result['first_token_ms']}ms")
            if result["error"]:
                print(f"       ERROR: {result['error'][:100]}")
            results.append(result)

    # 生成报告
    report = build_report(results, label)
    report["by_category"] = calc_category(report)

    # 保存
    os.makedirs(HISTORY_DIR, exist_ok=True)
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    history_file = os.path.join(HISTORY_DIR, f"{label}.json")
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print_report(report)
    print(f"\n  报告: {REPORT_FILE}")
    print(f"  历史: {history_file}")

    # 与基线对比
    if os.path.exists(BASELINE_FILE):
        with open(BASELINE_FILE, "r", encoding="utf-8") as f:
            baseline = json.load(f)
        print_diff(report, baseline)

    # 如果指定了 --save-baseline，存为基线
    if "--save-baseline" in sys.argv:
        with open(BASELINE_FILE, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"  ✅ 已保存为基线: {BASELINE_FILE}")


if __name__ == "__main__":
    asyncio.run(main())
