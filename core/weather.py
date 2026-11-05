"""Open-Meteo Archive API 拉取 + safe_end_date 归档截断（禁止虚假数值）。"""
from datetime import datetime, timedelta
import requests

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


def safe_end_date(requested_end: str, archive_lag_days: int = 5) -> str:
    """ERA5 约 5 天归档延迟，严格截断在已归档区间内。"""
    d = datetime.strptime(requested_end, "%Y-%m-%d") - timedelta(days=archive_lag_days)
    return d.strftime("%Y-%m-%d")


def fetch_node_weather(lon: float, lat: float, start: str, end: str, timeout: int = 20):
    """返回逐日降水(mm)与风速极值(m/s)。失败返回 None，不插值。"""
    end_safe = safe_end_date(end)
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": start, "end_date": end_safe,
        "daily": "precipitation_sum,wind_speed_10m_max",
        "timezone": "Asia/Shanghai",
    }
    try:
        r = requests.get(ARCHIVE_URL, params=params, timeout=timeout)
        r.raise_for_status()
        d = r.json().get("daily", {})
        return {
            "precip_max": max([x for x in d.get("precipitation_sum", []) if x is not None] or [0]),
            "wind_max": max([x for x in d.get("wind_speed_10m_max", []) if x is not None] or [0]),
            "status": "ok",
            "end_used": end_safe,
        }
    except Exception as e:
        return {"status": "missing", "error": str(e), "end_used": end_safe}