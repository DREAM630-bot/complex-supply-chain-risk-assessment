"""前端多智能体：Extractor → Feedback → Revision
四维硬规则：R1 证据一致性硬门禁 / R2 范围词校准 / R3 时效衰减 / R4 单因子一致性
"""
import re
from datetime import datetime

SCOPE_CAP = {"局部": 0.70, "区域": 0.85, "全局": 1.00}


def _norm(s: str) -> str:
    """去空格、统一标点、数值格式归一化。"""
    if s is None:
        return ""
    s = str(s)
    s = s.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    s = re.sub(r"[，,。；;！!？?、\s]", "", s)
    return s


def extractor(event: dict) -> dict:
    """规则版抽取器：产出 evidence_anchor / scope_word / severity_raw。"""
    text = event.get("text", "")
    anchor = event.get("anchor") or text[: min(30, len(text))]
    return {
        "id": event["id"],
        "node": event["node"],
        "date": event["date"],
        "title": event.get("title", ""),
        "evidence_anchor": anchor,
        "scope_word": event.get("scope", "局部"),
        "severity_raw": float(event.get("magnitude", 0.3)),
        "source": event.get("source", "公开报道"),
        "_text": text,
    }


def feedback(ext: dict, window_end: str) -> dict:
    """四维规则校验，返回带 rule_log 与最终分的记录。"""
    log = []
    score = ext["severity_raw"]
    text = ext["_text"]

    # R1 证据一致性硬门禁
    if _norm(ext["evidence_anchor"]) not in _norm(text):
        return {**ext, "severity_final": None, "rejected": True,
                "rule_log": ["R1 FAIL：证据锚点未在原文逐字存在，驳回"]}

    log.append("R1 PASS：证据锚点逐字命中")

    # R2 范围词校准
    cap = SCOPE_CAP.get(ext["scope_word"], 0.70)
    if score > cap:
        log.append(f"R2 触发：{ext['scope_word']}上限 {cap}，{score:.2f}→{cap:.2f}")
        score = cap
    else:
        log.append(f"R2 PASS：{ext['scope_word']}上限 {cap}")

    # R3 时效衰减（> 6 个月扣分，结构性风险豁免）
    try:
        d = datetime.strptime(ext["date"], "%Y-%m-%d")
        w = datetime.strptime(window_end, "%Y-%m-%d")
        months = (w.year - d.year) * 12 + (w.month - d.month)
        if months > 6:
            delta = -0.20 if months > 9 else -0.10
            log.append(f"R3 触发：距窗口终点 {months} 个月，{delta:+.2f}")
            score += delta
        else:
            log.append("R3 PASS：窗口内 6 个月时效有效")
    except Exception:
        log.append("R3 SKIP：日期解析失败")

    # R4 单因子一致性
    nums = re.findall(r"\d+(?:\.\d+)?", text)
    if len(set(nums)) <= 1 and ext["severity_raw"] > 0.60:
        delta = -0.15 if ext["severity_raw"] > 0.75 else -0.05
        log.append(f"R4 触发：单一数据点驱动高分，{delta:+.2f}")
        score += delta
    else:
        log.append("R4 PASS：多因子支撑")

    return {**ext, "severity_final": round(min(max(score, 0.0), 1.0), 4),
            "rejected": False, "rule_log": log}


def revision(fb: dict) -> dict:
    """有限修正：score_delta 下限 -0.30，clamp(0,1)。"""
    if fb.get("rejected"):
        return fb
    raw = fb["severity_raw"]
    final = fb["severity_final"]
    delta = max(final - raw, -0.30)
    corrected = round(min(max(raw + delta, 0.0), 1.0), 4)
    return {**fb, "severity_final": corrected, "score_delta": round(delta, 4)}


def run_pipeline(events: list, window_end: str) -> list:
    out = []
    for ev in events:
        ext = extractor(ev)
        fb = feedback(ext, window_end)
        out.append(revision(fb))
    return out


def node_exposure(results: list) -> dict:
    """节点暴露度 = 概率并集 1 - Π(1 - s_i)。"""
    exp = {}
    for r in results:
        if r.get("rejected") or r.get("severity_final") is None:
            continue
        n = r["node"]
        exp[n] = 1.0 - (1.0 - exp.get(n, 0.0)) * (1.0 - r["severity_final"])
    return exp