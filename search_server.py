from mcp.server.fastmcp import FastMCP
import httpx
from bs4 import BeautifulSoup
from knowledge_base import search as kb_search, add_document as kb_add
from skill_library import search_skills as sl_search, add_skill as sl_add

mcp = FastMCP("SearchServer")

SEARCH_URL = "https://cn.bing.com/search"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "zh-CN,zh;q=0.9",
}


@mcp.tool()
async def web_search(query: str, max_results: int = 5) -> str:
    """
    联网搜索，获取最新信息。用于查旅游攻略、景点推荐、美食、实时资讯等。
    :param query: 搜索关键词
    :param max_results: 返回结果数量（默认5条，最多10条）
    :return: 搜索结果
    """
    max_results = min(max_results, 10)

    try:
        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
            resp = await client.get(SEARCH_URL, params={"q": query, "count": max_results}, headers=HEADERS)
            resp.encoding = "utf-8"

        if resp.status_code != 200:
            return f"搜索失败: HTTP {resp.status_code}"

        soup = BeautifulSoup(resp.text, "html.parser")
        results = []

        for item in soup.select("li.b_algo"):
            title_el = item.select_one("h2 a")
            if not title_el:
                continue
            title = title_el.get_text(strip=True)
            url = title_el.get("href", "")

            snippet_el = item.select_one(".b_caption p, .b_lineclamp2, .b_lineclamp3")
            snippet = ""
            if snippet_el:
                snippet = snippet_el.get_text(strip=True)[:200]

            if title:
                results.append(f"标题: {title}\n链接: {url}\n摘要: {snippet}")

            if len(results) >= max_results:
                break

        if not results:
            return f"未找到关于「{query}」的搜索结果"

        return f"搜索「{query}」共找到 {len(results)} 条结果:\n\n" + "\n\n".join(results)

    except Exception as e:
        return f"搜索时出错: {str(e)}"


@mcp.tool()
async def fetch_url(url: str) -> str:
    """
    读取网页完整内容。用于深入阅读攻略、文章、博客等。
    :param url: 网页链接
    :return: 网页文本内容
    """
    try:
        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
            resp = await client.get(url, headers=HEADERS)

        if resp.status_code != 200:
            return f"获取网页失败: HTTP {resp.status_code}"

        # 尝试识别编码
        content_type = resp.headers.get("content-type", "")
        if "gb" in content_type.lower() or "gb2312" in content_type.lower():
            resp.encoding = "gbk"
        else:
            resp.encoding = resp.encoding or "utf-8"

        soup = BeautifulSoup(resp.text, "html.parser")

        # 移除无用元素
        for tag in soup.select("script, style, nav, footer, header, iframe, .sidebar, .comment, .ad, .advertisement"):
            tag.decompose()

        # 尝试提取正文区域
        main = soup.select_one("article, main, .content, .post-content, .article-content, .entry-content, #content")
        if not main:
            main = soup.select_one("body")
        if not main:
            return f"无法解析网页内容: {url}"

        # 提取文本
        text = main.get_text(separator="\n", strip=True)

        # 清理：合并多余空行
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        text = "\n".join(lines)

        # 限制长度
        if len(text) > 6000:
            text = text[:6000] + "\n\n...（内容过长，已截断前6000字）"

        if not text.strip():
            return f"网页内容为空或无法提取: {url}"

        # 自动入库
        try:
            page_title = soup.title.string.strip() if soup.title else url
            page_title = page_title[:100]
            kb_add(page_title, text[:6000])
        except Exception:
            pass

        return f"网页内容 ({url})，已自动保存到知识库:\n\n{text}"

    except Exception as e:
        return f"获取网页时出错: {str(e)}"


@mcp.tool()
async def search_knowledge(query: str) -> str:
    """
    搜索本地知识库（已保存的旅游攻略、文章等）。优先使用此工具，没有结果再联网搜索。
    :param query: 搜索关键词
    :return: 匹配的知识片段
    """
    try:
        results = kb_search(query, top_k=5)
        if not results:
            return "本地知识库中未找到相关内容"

        lines = [f"本地知识库找到 {len(results)} 条相关内容:\n"]
        for i, r in enumerate(results, 1):
            title = r.get("title", "未知")
            score = r.get("score", 0)
            content = r.get("content", "")[:300]
            lines.append(f"{i}. [{title}] (相关度: {score})\n{content}\n")
        return "\n".join(lines)
    except Exception as e:
        return f"搜索知识库时出错: {str(e)}"


@mcp.tool()
async def add_knowledge(title: str, content: str) -> str:
    """
    将文档添加到本地知识库。
    :param title: 文档标题
    :param content: 文档正文
    :return: 入库结果
    """
    try:
        n = kb_add(title, content)
        return f"已添加「{title}」到知识库，共 {n} 个语义块"
    except Exception as e:
        return f"添加知识失败: {str(e)}"


@mcp.tool()
async def search_skills(query: str) -> str:
    """
    搜索本地技能库（之前成功规划过的行程）。优先使用，命中后直接复用策略+已验证路线。
    :param query: 用户原始问题（如"成都3日游"）
    :return: 匹配的结构化技能（含策略+已验证路线）
    """
    try:
        results = sl_search(query, top_k=3)
        if not results:
            return "技能库中无匹配行程"

        lines = [f"技能库找到 {len(results)} 条相似行程:\n"]
        for i, r in enumerate(results, 1):
            s = r.get("strategy", {})
            lines.append(f"技能{i}：{r['query']} (相关度: {r['score']} | 城市: {r.get('city', '?')} | {r.get('days', 1)}天)")
            lines.append(f"访问{r.get('visit_count', 1)}次 | {r.get('created_at', '')}")
            if s.get("places"):
                lines.append(f"已验证景点: {', '.join(s['places'][:10])}")
            if s.get("proven_routes"):
                routes = [f"{p['from']}→{p['to']}" for p in s["proven_routes"][:5]]
                lines.append(f"已验证路线: {', '.join(routes)}")
            lines.append(f"参考行程:\n{r['itinerary'][:1500]}\n")
        return "\n".join(lines)
    except Exception as e:
        return f"搜索技能库出错: {str(e)}"
    except Exception as e:
        return f"搜索技能库出错: {str(e)}"


@mcp.tool()
async def add_skill(query: str, itinerary: str) -> str:
    """
    将成功规划的行程保存为技能。旅游规划完成后调用此工具，下次类似需求可直接复用。
    :param query: 用户原始问题
    :param itinerary: 完整行程内容
    :return: 保存结果
    """
    try:
        n = sl_add(query, itinerary)
        return f"技能已保存，当前技能库共 {n} 条"
    except Exception as e:
        return f"保存技能失败: {str(e)}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
