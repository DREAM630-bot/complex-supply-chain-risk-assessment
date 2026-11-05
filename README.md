# 复杂供应链风险全景测度系统

暴露度（文本测度）× 责任度（图博弈归因）二维交叉决策矩阵。

## 架构：读文字 / 算责任 / 做解释 三段硬解耦

| 段 | 模块 | 职责 |
|---|---|---|
| 离线·读文字 | `core/text_agents.py` | Extractor → Feedback(R1–R4) → Revision，输出可审计暴露度 |
| 离线·算责任 | `core/graph_engine.py` | 拆点有向图 + 多源多汇加权最大流 + 割集脆弱性 + Shapley |
| 在线·做解释 | `core/explainer.py` | DeepSeek 在强约束 JSON Context 下生成解释，含越界拦截 |

## 本地运行

```bash
pip install -r requirements.txt
export DEEPSEEK_API_KEY=sk-xxx
streamlit run app.py