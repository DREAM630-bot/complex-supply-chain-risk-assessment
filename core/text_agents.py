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
    # 兼容处理：把 events.json 里的字段映射到内部字典
    text = event.get("summary", "")  # 使用 summary 作为原文上下文
    
    # 锚点：优先用 JSON 里的 evidence_anchor，如果没有则截取正文
    anchor = event.get("evidence_anchor", "")
    if not anchor and text:
        anchor = text[: min(30, len(text))]
        
    # 节点：JSON 里是 node_ids 列表，这里统一转为列表
    nodes = event.get("node_ids") or [event.get("node")]
    
    return {
        "id": event.get("event_id") or event.get("id", "UNKNOWN"),
        "node": nodes,  # 注意：这里保留了列表形式，支持一对多
        "date": event.get("event_date") or event.get("date", "2026-01-01"),
        "title": event.get("event_type", "事件"), # 用 event_type 作标题
        "evidence_anchor": anchor,
        "scope_word": event.get("scope_word") or event.get("scope", "局部"),
        "severity_raw": float(event.get("severity_raw") or event.get("magnitude", 0.3)),
        "source": event.get("source", "公开报道"),
        "time_decay_exempt": event.get("time_decay_exempt", False),
        "_text": text + anchor,  # 将 summary 和 anchor 拼接作为校验文本
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

    # R3 时效衰减（> 6 个月扣分）
    if ext.get("time_decay_exempt"):
        log.append("R3 SKIP：结构性风险豁免时效衰减")
    else:
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
    """执行流水线，并将多节点事件拆分为独立记录以适配前端表格"""
    out = []
    for ev in events:
        ext = extractor(ev)
        fb = feedback(ext, window_end)
        rev = revision(fb)
        
        # 兼容处理：如果事件绑定了多个节点，将其拆解成多条记录
        nodes = rev["node"]
        if isinstance(nodes, list):
            for n in nodes:
                if not n:
                    continue
                new_rev = rev.copy()
                new_rev["node"] = n  # 强制降维成字符串
                out.append(new_rev)
        else:
            out.append(rev)
            
    return out


def node_exposure(results: list) -> dict:
    """节点暴露度 = 概率并集 1 - Π(1 - s_i)。"""
    exp = {}
    for r in results:
        if r.get("rejected") or r.get("severity_final") is None:
            continue
        
        n = r["node"]
        if not n:
            continue
        # 概率并集公式
        exp[n] = 1.0 - (1.0 - exp.get(n, 0.0)) * (1.0 - r["severity_final"])
        
    return exp
