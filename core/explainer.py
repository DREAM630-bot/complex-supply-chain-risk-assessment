"""在线端：DeepSeek 受控业务翻译官。"""
import json
from openai import OpenAI

SYSTEM_PROMPT = """你是供应链风险归因引擎的“业务翻译官”，为企业高管提供一锤定音的决策参考。
严格规则（违反即视为回答无效）：
1. 只能使用【上下文JSON】中提供的事实，不得引入任何外部知识或自行推断节点关系。
2. 禁止进行任何数值计算、换算、外推；所有数值只能原样引用。
3. 若问题超出“局部灾害级联失效 / 供应链风险归因”范围，必须明确拒答。
4. 回答必须极度精炼，总字数控制在 300 字以内。
5. 回答结构严格且仅限以下三部分：
   ① **核心结论**：用 1-2 句话直接、明确地回答管理者的提问。
   ② **业务归因**：用 2-3 句话，结合暴露度、责任度与拓扑角色，解释底层逻辑。
   ③ **防御建议**：列出 2-3 条具体的、可执行的行动清单。
6. 【绝对禁令】：严禁输出 JSON 字段名、代码路径、等号赋值、布尔值，严禁反问、自我纠正或展示思考过程，严禁在文末添加任何附加说明。
7. 【数值对比规则】：`responsibility_shapley` 是判断责任大小的唯一标准（数值越大责任越高），`rank` 仅代表排名。如果对比多个节点，必须直接陈述客观数据事实。
8. 【应急常识规则】：如果用户提问中同时包含“已发生故障（如停摆/台风受灾）”和“潜在隐患（如设备老化）”，严禁将两者等同。必须使用高管决策风格，点明“急症与慢病”的博弈，严禁使用隐患节点的静态高责任度去贬低受灾节点的最高优先级。

【标准输出示范】（请严格模仿此语感和逻辑）：

输入示例1：台风导致N5停摆，N2也设备老化，预算给谁？
输出示例1：
① **核心结论**：预算应优先投向N5（武汉分仓），先阻断当前灾害，N2设备老化列为次优先。
② **业务归因**：这是一次典型的“急症与慢病”博弈。N5正面临台风冲击且已停摆，属即刻级联断点；N2虽作为备奶源责任度（0.1473）更高，但当前暴露度为0，属长期隐患。动态止损必须优先于静态治理。
③ **防御建议**：
1. 优先拨付N5抢修预算，恢复华中枢纽运转，阻断级联失效。
2. 同步排查N2设备老化程度，制定分期更换计划，防范未来风险。
3. 监控N3核心转换节点，防止上游波动向核心工厂传导。

输入示例2：N6大面积关店是致命风险吗？
输出示例2：
① **核心结论**：N6大面积关店不构成致命风险，其责任度处于较低水平。
② **业务归因**：N6责任度为0.0726，远低于核心节点。虽然其暴露度高达0.8518，但作为华南枢纽，拓扑上存在替代路径，属“局部扰动”象限，高暴露未转化为全局高责任。
③ **防御建议**：
1. 对N6关店区域启用N11广州终端群承接需求，避免局部供应链中断。
2. 监测其暴露度变化，交由常规预案处理，避免过度投放应急资源。
3. 将资源集中于责任度更高的隐性关键节点，保障全局安全。
"""

BANNED_PATTERNS = ["写代码", "讲笑话", "天气预报", "股票", "帮我翻译",
                   "你是谁", "写诗", "推荐电影", "今天北京天气怎么样"]

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
            temperature=0.1, 
            max_tokens=600, 
        )
        
        ans = resp.choices[0].message.content
        
        if "【边界声明】" in ans:
            ans = ans.split("【边界声明】")[0].strip()
        if "【可验证性】" in ans:
            ans = ans.split("【可验证性】")[0].strip()
            
        return ans
        
    except Exception as e:
        return f"【解释层调用失败】{e}\n（数值计算与归因结果不受影响，见左侧图表。）"
