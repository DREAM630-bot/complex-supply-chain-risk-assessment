"""在线端：DeepSeek 受控业务翻译官。
强约束 JSON Context + 越界拦截 + 边界声明，不参与任何数值计算。
"""
import json
from openai import OpenAI

SYSTEM_PROMPT = """你是供应链风险归因引擎的“业务翻译官”。
严格规则（违反即视为回答无效）：
1. 只能使用【上下文JSON】中提供的事实，不得引入任何外部知识或自行推断节点关系。
2. 禁止进行任何数值计算、换算、外推；所有数值只能原样引用。
3. 若问题超出“局部灾害级联失效 / 供应链风险归因”范围，必须明确拒答。
4. 回答必须极度精炼，删去一切过渡性废话，总字数控制在 300 字以内。
5. 回答结构严格且仅限以下三部分（不要输出任何其他标题）：
   ① **核心结论**：用 1-2 句话直接、明确地回答管理者的提问（影响大不大？该不该优先投钱？）。
   ② **业务归因**：用 2-3 句话，结合暴露度、责任度与拓扑角色，解释“为什么得出这个结论”。必须点透它是“显性核心”或“隐性关键”的底层逻辑。例如：它平时没有负面预警（暴露度低），但作为核心备份路径，一旦主源中断，它将承担极大的兜底流量（责任度高），属于典型的潜伏炸弹。
   ③ **防御建议**：列出 2-3 条具体的、可执行的行动清单。
6. 【绝对禁令】：严禁输出任何 JSON 的英文字段名、代码路径、等号赋值或布尔值（如 node.role、scores.exposure、true 等）。严禁大段罗列数据或论证过程。
7. 每次回答末尾必须追加：【边界声明】本回答基于公开信息代理映射，不代表企业内部真实数据。
"""

BANNED_PATTERNS = ["写代码", "讲笑话", "天气预报", "股票", "帮我翻译",
                   "你是谁", "写诗", "推荐电影"]

OUT_OF_SCOPE = ["预测明年", "反推内部", "绝对货量", "企业机密", "精确产能"]


def guard(question: str):
    q = (question or "").strip()
    if len(q) < 2:
        return False, "问题过短，无法受理。"
    for b in BANNED_PATTERNS:
        if b in q:
            return False, "该问题超出供应链风险归因的受理范围，系统拒绝作答。"
    for o in OUT_OF_SCOPE:
        if o in q:
            return False, "系统不具备也不声称具备推断企业内部绝对数据的能力，请聚焦公开信息下的结构风险。"
    return True, ""


def build_context(node_row: dict, top_nodes: list) -> dict:
    """强约束 JSON Context：拓扑指标 + Shapley + 证据锚点 + 触发规则 + 精度等级。"""
    return {
        "node": {
            "id": node_row["id"], "name": node_row["name"], "type": node_row["type"],
            "role": node_row["role"], "critical_transition": node_row.get("critical", False),
            "capacity_source": node_row.get("capacity_source"),
            "topology": {
                "in_degree": node_row.get("in_degree"),
                "out_degree": node_row.get("out_degree"),
                "is_articulation_point": node_row.get("is_cut_point"),
                "is_bridge_endpoint": node_row.get("is_bridge"),
            },
        },
        "scores": {
            "exposure": node_row.get("exposure"),
            "responsibility_shapley": node_row.get("responsibility"),
            "rank": node_row.get("rank"),
            "quadrant": node_row.get("quadrant"),
        },
        "evidence_anchors": node_row.get("anchors", []),
        "rule_log": node_row.get("rule_log", []),
        "trigger_rules": ["R1证据一致性", "R2范围词校准", "R3时效衰减", "R4单因子一致性"],
        "network_context": {
            "top_k_responsibility": top_nodes,
            "demand_weight": node_row.get("demand_weight"),
            "penalty_params": node_row.get("penalty"),
        },
        "data_boundary": "所有容量为公开锚点/相对比例/业务先验三级精度映射",
    }


def explain(api_key: str, context: dict, question: str,
            model: str = "deepseek-chat", timeout: int = 60) -> str:
    ok, reason = guard(question)
    if not ok:
        return f"【拒答】{reason}"

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com", timeout=timeout)
    user_msg = (
        "【上下文JSON】\n"
        + json.dumps(context, ensure_ascii=False, indent=2)
        + "\n\n【管理者问题】\n"
        + question
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.2,
            max_tokens=600, 
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"【解释层调用失败】{e}\n（数值计算与归因结果不受影响，见左侧图表。）"
