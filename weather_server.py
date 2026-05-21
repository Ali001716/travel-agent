from mcp.server.fastmcp import FastMCP
import httpx

mcp = FastMCP("WeatherServer")

# WMO 天气代码 → 中文描述
WMO_CODES = {
    0: "晴天", 1: "大部晴朗", 2: "多云", 3: "阴天",
    45: "有雾", 48: "雾凇",
    51: "小毛毛雨", 53: "毛毛雨", 55: "大毛毛雨",
    61: "小雨", 63: "中雨", 65: "大雨",
    71: "小雪", 73: "中雪", 75: "大雪",
    80: "小阵雨", 81: "中阵雨", 82: "大阵雨",
    95: "雷暴", 96: "冰雹雷暴", 99: "强冰雹雷暴",
}


@mcp.tool()
async def query_weather(city: str) -> str:
    """
    使用 Open-Meteo API 查询全球城市天气（免费，无需 API Key）。
    :param city: 城市名称（支持中英文，如"北京"、"Tokyo"、"London"）
    """
    async with httpx.AsyncClient() as client:
        try:
            # Step 1: 地理编码
            geo_url = "https://geocoding-api.open-meteo.com/v1/search"
            geo_resp = await client.get(geo_url, params={
                "name": city, "count": 1, "language": "zh"
            })
            geo_data = geo_resp.json()

            if not geo_data.get("results"):
                return f"未找到城市: {city}"

            result = geo_data["results"][0]
            lat = result["latitude"]
            lng = result["longitude"]
            city_name = result.get("name", city)
            country = result.get("country", "")

            # Step 2: 获取天气
            weather_url = "https://api.open-meteo.com/v1/forecast"
            weather_resp = await client.get(weather_url, params={
                "latitude": lat,
                "longitude": lng,
                "current": "temperature_2m,relative_humidity_2m,wind_direction_10m,wind_speed_10m,weather_code",
            })
            weather_data = weather_resp.json()
            current = weather_data["current"]

            weather_code = current["weather_code"]
            weather_text = WMO_CODES.get(weather_code, f"未知({weather_code})")

            # 风向角度转文字
            wind_deg = current["wind_direction_10m"]
            wind_dirs = ["北", "东北", "东", "东南", "南", "西南", "西", "西北"]
            wind_text = wind_dirs[round(wind_deg / 45) % 8]

            return f"""
城市: {city_name}{'，' + country if country else ''}
天气: {weather_text}
温度: {current['temperature_2m']}°C
湿度: {current['relative_humidity_2m']}%
风速: {current['wind_speed_10m']} km/h
风向: {wind_text}
            """.strip()

        except Exception as e:
            return f"查询天气时出错: {str(e)}"


@mcp.tool()
async def get_weather_tips(season: str) -> str:
    """获取指定季节的天气贴士"""
    tips = {
        "spring": "春季多风，注意防风保暖，早晚温差大请及时添衣",
        "summer": "夏季炎热，注意防暑防晒，多喝水",
        "autumn": "秋季干燥，注意润肺防燥，适当增减衣物",
        "winter": "冬季寒冷，注意防寒保暖，小心感冒",
    }
    return tips.get(season.lower(), "未知季节，请输入 spring/summer/autumn/winter")


@mcp.tool()
async def forecast_weather(city: str, days: int = 5) -> str:
    """
    查询未来多日天气预报（免费，无需API Key）。
    :param city: 城市名称（支持中英文）
    :param days: 预报天数（1-7天，默认5天）
    :return: 未来几天的天气预报
    """
    days = min(max(days, 1), 7)

    async with httpx.AsyncClient() as client:
        try:
            # 地理编码
            geo_resp = await client.get(
                "https://geocoding-api.open-meteo.com/v1/search",
                params={"name": city, "count": 1, "language": "zh"},
            )
            geo_data = geo_resp.json()
            if not geo_data.get("results"):
                return f"未找到城市: {city}"

            result = geo_data["results"][0]
            lat = result["latitude"]
            lng = result["longitude"]
            city_name = result.get("name", city)
            country = result.get("country", "")

            # 获取逐日预报
            weather_resp = await client.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": lat,
                    "longitude": lng,
                    "daily": "temperature_2m_max,temperature_2m_min,weather_code,precipitation_probability_max",
                    "forecast_days": days,
                    "timezone": "auto",
                },
            )
            weather_data = weather_resp.json()
            daily = weather_data["daily"]

            lines = [f"城市: {city_name}{'，' + country if country else ''}"]
            lines.append(f"未来{days}天天气预报:\n")

            for i in range(len(daily["time"])):
                date = daily["time"][i]
                max_t = daily["temperature_2m_max"][i]
                min_t = daily["temperature_2m_min"][i]
                code = daily["weather_code"][i]
                rain = daily["precipitation_probability_max"][i]
                weather = WMO_CODES.get(code, f"未知({code})")

                lines.append(
                    f"{date}: {weather} | {min_t}°C ~ {max_t}°C | 降雨概率{rain}%"
                )

            return "\n".join(lines)

        except Exception as e:
            return f"查询天气预报时出错: {str(e)}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
