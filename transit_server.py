import json
import os
import urllib.parse
import uuid

import httpx
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

load_dotenv()

mcp = FastMCP("TransitServer")

AMAP_KEY = os.getenv("AMAP_API_KEY", "")
GEO_URL = "https://restapi.amap.com/v3/geocode/geo"
PLACE_URL = "https://restapi.amap.com/v3/place/text"
TRANSIT_URL = "https://restapi.amap.com/v3/direction/transit/integrated"
DRIVING_URL = "https://restapi.amap.com/v3/direction/driving"
WALKING_URL = "https://restapi.amap.com/v3/direction/walking"


def to_int(val, default=0):
    if isinstance(val, list):
        val = val[0] if val else default
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


async def geocode(address: str, city: str = "") -> tuple[str, str] | None:
    """
    地址/地名 → 经纬度。
    先尝试 POI 搜索（适合商场、景点等），失败则回退到地理编码（适合地址）。
    """
    async with httpx.AsyncClient() as client:
        # 1. POI 搜索（适合地名：商场、景点、小区等）
        poi_params = {"key": AMAP_KEY, "keywords": address}
        if city:
            poi_params["city"] = city
        poi_resp = await client.get(PLACE_URL, params=poi_params)
        poi_data = poi_resp.json()
        if poi_data.get("status") == "1" and poi_data.get("pois"):
            poi = poi_data["pois"][0]
            return (poi["location"], poi.get("name", address))

        # 2. 回退：地理编码（适合正式地址）
        geo_params = {"key": AMAP_KEY, "address": address}
        if city:
            geo_params["city"] = city
        geo_resp = await client.get(GEO_URL, params=geo_params)
        geo_data = geo_resp.json()
        if geo_data.get("status") == "1" and geo_data.get("geocodes"):
            geo = geo_data["geocodes"][0]
            return (geo["location"], geo.get("formatted_address", address))

        return None


def format_duration(seconds: int) -> str:
    """秒数 → 人类可读的时间"""
    if seconds < 60:
        return f"{seconds}秒"
    elif seconds < 3600:
        return f"{seconds // 60}分钟"
    else:
        h = seconds // 3600
        m = (seconds % 3600) // 60
        return f"{h}小时{m}分钟" if m else f"{h}小时"


@mcp.tool()
async def search_transit(from_place: str, to_place: str, city: str = "") -> str:
    """
    查询公交/地铁出行路线方案。
    :param from_place: 出发地（地址或地名）
    :param to_place: 目的地（地址或地名）
    :param city: 城市（必填，如"广州"）
    """
    if not AMAP_KEY:
        return "错误：未配置高德地图 API Key"

    try:
        from_result = await geocode(from_place, city)
        to_result = await geocode(to_place, city)
        if not from_result:
            return f"找不到出发地: {from_place}"
        if not to_result:
            return f"找不到目的地: {to_place}"
        from_coord, from_addr = from_result
        to_coord, to_addr = to_result

        params = {
            "key": AMAP_KEY, "origin": from_coord, "destination": to_coord,
            "city": city or "", "strategy": "0", "show_fields": "cost",
        }
        async with httpx.AsyncClient() as client:
            resp = await client.get(TRANSIT_URL, params=params)
            data = resp.json()

        if data.get("status") != "1":
            return f"公交路线查询失败: {data.get('info', '未知错误')}"

        routes = data.get("route", {}).get("transits", [])
        if not routes:
            return f"从「{from_addr}」到「{to_addr}」没有找到公交/地铁方案"

        lines = [f"公交/地铁 — 从「{from_addr}」到「{to_addr}」共 {len(routes)} 个方案:\n"]
        for i, route in enumerate(routes[:3], 1):
            duration = to_int(route.get("duration"))
            walking = to_int(route.get("walking_distance"))
            cost_str = ""
            if "cost" in route:
                c = route["cost"]
                if isinstance(c, list):
                    c = c[0] if c else 0
                if isinstance(c, str):
                    c = float(c)
                if c:
                    cost_str = f" | 约{int(c)}元"
            lines.append(f"方案{i}：{format_duration(duration)} | 步行{walking}米{cost_str}")

            for seg in route.get("segments", []):
                if not isinstance(seg, dict):
                    continue
                bus_data = seg.get("bus", {})
                if isinstance(bus_data, dict):
                    for bus in bus_data.get("buslines", []):
                        if not isinstance(bus, dict):
                            continue
                        name = bus.get("name", "未知线路")
                        dep = bus.get("departure_stop", {}) or {}
                        arr = bus.get("arrival_stop", {}) or {}
                        dep = dep.get("name", "?") if isinstance(dep, dict) else str(dep)
                        arr = arr.get("name", "?") if isinstance(arr, dict) else str(arr)
                        via = to_int(bus.get("via_num"))
                        lines.append(f"  {name}（{dep} → {arr}，{via}站）")
                walk = seg.get("walking", {})
                if isinstance(walk, dict):
                    wd = to_int(walk.get("distance"))
                    if wd > 100:
                        lines.append(f"  步行{wd}米")
            lines.append("")

        return "\n".join(lines).strip()
    except Exception as e:
        return f"查询公交路线时出错: {str(e)}"


@mcp.tool()
async def search_driving(from_place: str, to_place: str, city: str = "") -> str:
    """
    查询驾车路线（含实时路况）。
    :param from_place: 出发地（地址或地名）
    :param to_place: 目的地（地址或地名）
    :param city: 城市
    """
    if not AMAP_KEY:
        return "错误：未配置高德地图 API Key"

    try:
        from_result = await geocode(from_place, city)
        to_result = await geocode(to_place, city)
        if not from_result:
            return f"找不到出发地: {from_place}"
        if not to_result:
            return f"找不到目的地: {to_place}"
        from_coord, from_addr = from_result
        to_coord, to_addr = to_result

        params = {
            "key": AMAP_KEY, "origin": from_coord, "destination": to_coord,
            "extensions": "all", "show_fields": "cost",
        }
        async with httpx.AsyncClient() as client:
            resp = await client.get(DRIVING_URL, params=params)
            data = resp.json()

        if data.get("status") != "1":
            return f"驾车路线查询失败: {data.get('info', '未知错误')}"

        paths = data.get("route", {}).get("paths", [])
        if not paths:
            return f"从「{from_addr}」到「{to_addr}」没有找到驾车方案"

        lines = [f"驾车 — 从「{from_addr}」到「{to_addr}」共 {len(paths)} 个方案:\n"]
        for i, path in enumerate(paths[:3], 1):
            distance = to_int(path.get("distance"))
            duration = to_int(path.get("duration"))

            # 路况信息
            traffic_info = ""
            if "traffic" in path:
                pass  # 高德免费版不返回详细路况字段，但仍可基于 duration 做判断

            # 过路费
            tolls = to_int(path.get("tolls"))
            toll_str = ""
            if tolls > 0:
                toll_str = f" | 过路费约{tolls}元"
            elif "tolls" in path and tolls == 0:
                toll_str = " | 无过路费"

            lines.append(f"方案{i}：{format_duration(duration)} | {distance / 1000:.1f}公里{toll_str}")

            # 关键路段
            steps = path.get("steps", [])
            for step in steps[:6]:  # 只取前 6 段关键导航
                if not isinstance(step, dict):
                    continue
                instruction = step.get("instruction", "")
                road = step.get("road", "")
                if instruction:
                    lines.append(f"  {instruction}")
            if len(steps) > 6:
                lines.append(f"  ... 共 {len(steps)} 段导航")
            lines.append("")

        return "\n".join(lines).strip()
    except Exception as e:
        return f"查询驾车路线时出错: {str(e)}"


@mcp.tool()
async def search_walking(from_place: str, to_place: str, city: str = "") -> str:
    """
    查询步行路线。
    :param from_place: 出发地（地址或地名）
    :param to_place: 目的地（地址或地名）
    :param city: 城市
    """
    if not AMAP_KEY:
        return "错误：未配置高德地图 API Key"

    try:
        from_result = await geocode(from_place, city)
        to_result = await geocode(to_place, city)
        if not from_result:
            return f"找不到出发地: {from_place}"
        if not to_result:
            return f"找不到目的地: {to_place}"
        from_coord, from_addr = from_result
        to_coord, to_addr = to_result

        params = {
            "key": AMAP_KEY, "origin": from_coord, "destination": to_coord,
        }
        async with httpx.AsyncClient() as client:
            resp = await client.get(WALKING_URL, params=params)
            data = resp.json()

        if data.get("status") != "1":
            return f"步行路线查询失败: {data.get('info', '未知错误')}"

        paths = data.get("route", {}).get("paths", [])
        if not paths:
            return f"从「{from_addr}」到「{to_addr}」没有找到步行方案"

        lines = [f"步行 — 从「{from_addr}」到「{to_addr}」共 {len(paths)} 个方案:\n"]
        for i, path in enumerate(paths[:3], 1):
            distance = to_int(path.get("distance"))
            duration = to_int(path.get("duration"))
            lines.append(f"方案{i}：{format_duration(duration)} | {distance}米")

            steps = path.get("steps", [])
            for step in steps[:5]:
                if not isinstance(step, dict):
                    continue
                instruction = step.get("instruction", "")
                if instruction:
                    lines.append(f"  {instruction}")
            if len(steps) > 5:
                lines.append(f"  ... 共 {len(steps)} 段")
            lines.append("")

        return "\n".join(lines).strip()
    except Exception as e:
        return f"查询步行路线时出错: {str(e)}"


@mcp.tool()
async def search_places(keyword: str, city: str = "", max_results: int = 10) -> str:
    """
    搜索指定类型的店铺或场所（密室逃脱、剧本杀、咖啡店、餐厅等）。
    :param keyword: 搜索关键词，如"密室逃脱"、"日料"、"剧本杀"
    :param city: 城市（如"广州"）
    :param max_results: 返回数量（最多15条）
    :return: 搜索结果列表
    """
    if not AMAP_KEY:
        return "错误：未配置高德地图 API Key"

    max_results = min(max_results, 15)

    try:
        params = {"key": AMAP_KEY, "keywords": keyword, "offset": max_results}
        if city:
            params["city"] = city

        async with httpx.AsyncClient() as client:
            resp = await client.get(PLACE_URL, params=params)
            data = resp.json()

        if data.get("status") != "1":
            return f"搜索失败: {data.get('info', '未知错误')}"

        pois = data.get("pois", [])
        if not pois:
            return f"在{city or '全国'}未找到「{keyword}」"

        # 存到 JSON 文件供前端获取坐标
        sid = str(uuid.uuid4())[:8]
        items = []
        for p in pois:
            items.append({
                "name": p.get("name", "?"),
                "addr": p.get("address", ""),
                "lng": p.get("location", "").split(",")[0] if p.get("location") else "",
                "lat": p.get("location", "").split(",")[1] if p.get("location") else "",
                "rating": p.get("biz_ext", {}).get("rating", ""),
                "cost": p.get("biz_ext", {}).get("cost", ""),
            })
        with open("static/search_cache.json", "w", encoding="utf-8") as f:
            json.dump({sid: items}, f, ensure_ascii=False)

        lines = [f"搜索「{keyword}」共找到 {len(pois)} 个 {{search_id:{sid}}}\n"]
        for i, p in enumerate(pois, 1):
            name = p.get("name", "?")
            addr = p.get("address", "?")
            rating = p.get("biz_ext", {}).get("rating", "") or ""
            cost = p.get("biz_ext", {}).get("cost", "") or ""
            tel = p.get("tel", "") or ""

            extras = []
            if rating:
                extras.append(f"评分{rating}")
            if cost:
                extras.append(f"人均{cost}")
            if tel:
                extras.append(tel)

            line = f"{i}. **{name}**"
            if addr:
                line += f" | {addr}"
            if extras:
                line += f" | {' · '.join(extras)}"
            lines.append(line)

        return "\n".join(lines)

    except Exception as e:
        return f"搜索店铺时出错: {str(e)}"


@mcp.tool()
async def search_nearby(keyword: str, lat: float, lng: float, radius: int = 3000) -> str:
    """
    搜索用户附近的店铺或场所。
    :param keyword: 搜索关键词
    :param lat: 当前纬度
    :param lng: 当前经度
    :param radius: 搜索半径（米，默认3000米）
    :return: 附近搜索结果
    """
    if not AMAP_KEY:
        return "错误：未配置高德地图 API Key"

    try:
        params = {
            "key": AMAP_KEY,
            "keywords": keyword,
            "location": f"{lng},{lat}",
            "radius": min(radius, 10000),
            "sortrule": "distance",
            "offset": 10,
        }

        async with httpx.AsyncClient() as client:
            resp = await client.get("https://restapi.amap.com/v3/place/around", params=params)
            data = resp.json()

        if data.get("status") != "1":
            return f"搜索失败: {data.get('info', '未知错误')}"

        pois = data.get("pois", [])
        if not pois:
            return f"附近{radius}米内未找到「{keyword}」"

        sid = str(uuid.uuid4())[:8]
        items = []
        for p in pois:
            items.append({
                "name": p.get("name", "?"),
                "addr": p.get("address", ""),
                "lng": p.get("location", "").split(",")[0] if p.get("location") else "",
                "lat": p.get("location", "").split(",")[1] if p.get("location") else "",
                "distance": p.get("distance", ""),
                "rating": p.get("biz_ext", {}).get("rating", ""),
            })
        with open("static/search_cache.json", "w", encoding="utf-8") as f:
            json.dump({sid: items}, f, ensure_ascii=False)

        lines = [f"附近「{keyword}」共找到 {len(pois)} 个 {{search_id:{sid}}}\n"]
        for i, p in enumerate(pois, 1):
            name = p.get("name", "?")
            addr = p.get("address", "?")
            distance = p.get("distance", "?")
            rating = p.get("biz_ext", {}).get("rating", "") or ""

            line = f"{i}. **{name}**"
            if distance and distance != "?":
                line += f" | {distance}米"
            if addr:
                line += f" | {addr}"
            if rating:
                line += f" | 评分{rating}"
            lines.append(line)

        return "\n".join(lines)

    except Exception as e:
        return f"搜索附近店铺时出错: {str(e)}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
