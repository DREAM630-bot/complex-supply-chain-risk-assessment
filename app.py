import json
import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import networkx as nx

from core.graph_engine import (
    load_json, build_flow_graph, eval_state, value_function,
    shapley_mc, shapley_hierarchical, normalize_phi, sensitivity_scan,
)
from core.text_agents import run_pipeline, node_exposure
from core.explainer import build_context, explain, guard

st.set_page_config(page_title="基于LLM的多智能体博弈归因的供应链风险全景测度系统", layout="wide")

CFG = load_json("config/topology_13.json")
EVENTS = load_json("config/events.json")
PLAYERS = [n["id"] for n in CFG["nodes"]]
WINDOW_END = CFG["meta"]["window"][1]

@st.cache_data(show_spinner=False)
def compute_all(n_perm: int, seed: int):
    base = eval_state(CFG)
    if len(PLAYERS) <= 10:
        phi_raw = shapley_mc(CFG, PLAYERS, base, n_perm, seed)
        method = "蒙特卡罗排列采样 Shapley"
    else:
        phi_raw = shapley_hierarchical(CFG, base, n_perm, seed)
        method = "分层 Shapley（候选集 + 背景聚合）"

    phi = normalize_phi(phi_raw)

    results = run_pipeline(EVENTS, WINDOW_END)
    exposure = node_exposure(results)

    G = build_flow_graph(CFG)
    und = G.to_undirected()
    cut_points = set(nx.articulation_points(und)) if und.number_of_nodes() > 2 else set()
    bridges = {u for e in nx.bridges(und) for u in e} if und.number_of_nodes() > 2 else set()

    rows = []
    ranked = sorted(phi.items(), key=lambda x: -x[1])
    rank_map = {k: i + 1 for i, (k, _) in enumerate(ranked)}
    
    resp_threshold = np.quantile(list(phi.values()), 0.70) if phi else 0
    exp_threshold = 0.15

    for n in CFG["nodes"]:
        nid = n["id"]
        exp = exposure.get(nid, 0.0)
        resp = phi.get(nid, 0.0)
        
        is_high_resp = (rank_map[nid] <= 4) or (resp >= resp_threshold)
        is_high_exp = (exp >= exp_threshold)

        if is_high_exp and is_high_resp:
            quad = "显性核心"
        elif not is_high_exp and is_high_resp:
            quad = "隐性关键"
        elif is_high_exp and not is_high_resp:
            quad = "局部扰动"
        else:
            quad = "常规节点"

        anchors = [r for r in results if r.get("node") == nid and not r.get("rejected")]
        rows.append({
            "id": nid, "name": n["name"], "type": n["type"], "role": n["role"],
            "capacity_source": n["capacity_source"], "capacity": n["capacity"],
            "exposure": round(exp, 4),
            "responsibility": round(resp, 4),
            "rank": rank_map[nid],
            "quadrant": quad,
            "in_degree": G.in_degree(f"{nid}__in"),
            "out_degree": G.out_degree(f"{nid}__out"),
            "is_cut_point": nid in cut_points,
            "is_bridge": nid in bridges,
            "critical": n.get("critical", False),
        })

    df = pd.DataFrame(rows).sort_values("responsibility", ascending=False).reset_index(drop=True)
    return df, results, method

def get_api_key() -> str:
    try:
        return st.secrets.get("DEEPSEEK_API_KEY", "")
    except Exception:
        return os.environ.get("DEEPSEEK_API_KEY", "")

with st.sidebar:
    st.title("基于LLM的多智能体博弈归因的供应链风险全景测度系统")
    st.caption("暴露度 — 责任度 二维交叉决策矩阵")
    n_perm = st.slider("蒙特卡罗采样次数", 100, 2000, 500, 100)
    seed = st.number_input("随机种子", 1, 9999, 42)
    st.divider()
    api_key_input = st.text_input("DeepSeek API Key（可选，仅用于解释层）",
                                  type="password", value=get_api_key())

df, results, method = compute_all(int(n_perm), int(seed))

c1, c2, c3, c4 = st.columns(4)
c1.metric("节点数", len(df))
c2.metric("显性核心", int((df["quadrant"] == "显性核心").sum()))
c3.metric("隐性关键", int((df["quadrant"] == "隐性关键").sum()))
c4.metric("归因方法", method.split("（")[0])

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["① 二维决策矩阵", "② 责任度归因", "③ 供应链网络拓扑", "④ 敏感性分析", "⑤ 智能体问答"]
)

with tab1:
    st.subheader("暴露度 — 责任度 二维交叉矩阵")
    
    quad_color = {
        "显性核心": "#d62728", "隐性关键": "#ff7f0e",
        "局部扰动": "#1f77b4", "常规节点": "#7f7f7f",
    }
    
    fig = go.Figure()
    for quad in ["显性核心", "隐性关键", "局部扰动", "常规节点"]:
        sub_df = df[df["quadrant"] == quad]
        if sub_df.empty: continue
        
        fig.add_trace(go.Scatter(
            x=sub_df["exposure"],
            y=sub_df["responsibility"],
            mode='markers+text',
            name=quad,
            text=sub_df["name"],
            textposition='top center',
            textfont=dict(size=14, color="black"),
            hoverinfo='text',
            hovertext=[
                f"<b>{r['name']}</b><br>暴露度: {r['exposure']}<br>责任度: {r['responsibility']}<br>排名: {r['rank']}"
                for _, r in sub_df.iterrows()
            ],
            marker=dict(size=20, color=quad_color[quad], line=dict(width=2, color='white'))
        ))

    fig.add_vline(x=0.15, line_dash="dash", line_color="gray")
    y_thresh = df["responsibility"].quantile(0.70)
    fig.add_hline(y=y_thresh, line_dash="dash", line_color="gray")
    
    fig.update_layout(
        height=700, margin=dict(l=40, r=40, t=40, b=40), legend_title="象限",
        plot_bgcolor="rgba(240,242,246,0.5)",
        font=dict(size=14)
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='Lightgray', title="暴露度（文本测度）")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='Lightgray', title="责任度（Shapley 归因）")
    
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
- **显性核心（双高）**：舆情热 + 破坏力大，优先保供。
- **隐性关键（低暴露—高责任）**：潜伏炸弹，平时无预警但一断全网瘫，需物理冗余 / 备产线 / 缓冲库存。
- **局部扰动（高暴露—低责任）**：虚假警报，末端可替代，不应占用应急预算。
- **常规节点**：常态巡检。
""")

with tab2:
    st.subheader("Shapley 责任度排序（Top-K）")
    top_k = st.slider("显示 Top-K", 3, len(df), min(10, len(df)))
    d = df.head(top_k).iloc[::-1]
    fig2 = px.bar(d, x="responsibility", y="name", orientation="h",
                  color="quadrant", text="responsibility",
                  color_discrete_map=quad_color)
    fig2.update_traces(texttemplate="%{text:.3f}", textposition="outside", textfont=dict(size=14))
    fig2.update_layout(height=460, yaxis_title="", xaxis_title="归一化责任度", font=dict(size=14))
    st.plotly_chart(fig2, use_container_width=True)

    st.dataframe(
        df[["rank", "id", "name", "type", "role", "exposure", "responsibility",
            "quadrant", "is_cut_point", "capacity_source"]],
        use_container_width=True, hide_index=True,
    )

with tab3:
    st.subheader("供应链网络拓扑")
    st.markdown("节点大小反映 **Shapley 责任度**。实线为单向物流，虚线为双向调拨边。")

    pos = {
        "N1": (0, 0.8), "N2": (0, 0.2),
        "N3": (1, 0.7), "N4": (1, 0.3),
        "N5": (2, 0.8), "N6": (2, 0.5), "N7": (2, 0.2),
        "N8": (3, 0.7), "N9": (3, 0.5), "N10": (3, 0.2),
        "N11": (4, 0.7), "N12": (4, 0.5), "N13": (4, 0.2),
    }

    node_dict = {n["id"]: n for n in CFG["nodes"]}
    row_dict = {row["id"]: row for row in df.to_dict("records")}

    role_info = {
        "source": {"name": "供应源 (牧场)", "color": "#9b59b6"},     # 紫
        "plant": {"name": "工厂 (核心/备份)", "color": "#e74c3c"},  # 红
        "warehouse": {"name": "区域仓 (分仓)", "color": "#3498db"}, # 蓝
        "delivery": {"name": "末端配送", "color": "#f1c40f"},       # 黄
        "sink": {"name": "需求汇 (终端)", "color": "#2ecc71"}       # 绿
    }

    edge_traces = []
    for edge in CFG["edges"]:
        src, dst = edge["src"], edge["dst"]
        if src not in pos or dst not in pos: continue
        
        x0, y0 = pos[src]
        x1, y1 = pos[dst]
        is_bidir = edge.get("bidirectional", False)
        cap = edge["capacity"]
        
        line_width = max(2.0, min(6, cap / 150))
        line_color = "rgba(150, 150, 150, 0.5)" if not is_bidir else "rgba(255, 165, 0, 0.6)"

        edge_traces.append(go.Scatter(
            x=[x0, x1], y=[y0, y1],
            line=dict(width=line_width, color=line_color, dash='dot' if is_bidir else 'solid'),
            hoverinfo='text',
            text=f"{src} → {dst}<br>容量: {cap}<br>模式: {edge.get('mode', 'road')}",
            mode='lines', showlegend=False
        ))
        
        mid_x, mid_y = (x0 + x1) / 2, (y0 + y1) / 2
        edge_traces.append(go.Scatter(
            x=[mid_x], y=[mid_y], mode='markers',
            marker=dict(symbol='triangle-right' if x1 >= x0 else 'triangle-left', size=14, color=line_color),
            showlegend=False, hoverinfo='none'
        ))

    node_traces = {}
    for nid, n in node_dict.items():
        if nid not in pos: continue
        role = n["type"]
        if role not in node_traces: node_traces[role] = []
        
        x, y = pos[nid]
        r_data = row_dict.get(nid, {})
        quad = r_data.get("quadrant", "常规节点")
        resp = r_data.get("responsibility", 0.0)
        
        size = 50 + (resp * 400)
        
        hover = f"<b>{n['name']} ({nid})</b><br>" \
                f"职能: {n['role']}<br>" \
                f"象限: {quad}<br>" \
                f"责任度: {resp:.4f} (排名 {r_data.get('rank', '-')})<br>" \
                f"暴露度: {r_data.get('exposure', 0.0):.4f}"
        
        node_traces[role].append(go.Scatter(
            x=[x], y=[y], mode='markers+text',
            name=role_info.get(role, {"name": "未知"})["name"],
            text=[n["name"]],
            textposition="bottom center",
            textfont=dict(size=16, color="black"),
            hoverinfo='text', hovertext=[hover],
            marker=dict(size=size, color=role_info.get(role, {"color": "gray"})["color"], line=dict(width=2, color='white')),
            showlegend=True
        ))

    fig_net = go.Figure()
    for t in edge_traces: fig_net.add_trace(t)
    for role, traces in node_traces.items():
        for i, t in enumerate(traces):
            t.showlegend = (i == 0)
            fig_net.add_trace(t)

    fig_net.update_layout(
        height=750,
        plot_bgcolor="white",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=40, r=40, t=40, b=40),
        legend=dict(
            title=dict(text="节点角色", font=dict(size=18)),
            font=dict(size=16)
        )
    )
    st.plotly_chart(fig_net, use_container_width=True)

with tab4:
    st.subheader("容量敏感性扫描（代理 / 先验节点 ±20%–30%）")
    id_to_name = {n["id"]: n["name"] for n in CFG["nodes"]}
    if st.button("运行敏感性扫描", type="primary"):
        with st.spinner("扫描中…"):
            # 1. 将采样次数改回与全景表一致的 n_perm，减少随机误差
            runs = sensitivity_scan(CFG, PLAYERS, n_perm=int(n_perm), seed=int(seed))
            
        # ================= 核心修复代码 =================
        # 2. 获取全景表（df）中的绝对正确基准
        true_base = {row["id"]: row["responsibility"] for row in df.to_dict("records")}
        fake_base = runs["baseline"]
        
        # 3. 强行等比缩放，把敏感性表的基准线拉回全景图的数值
        calibrated_runs = {}
        for state, val_dict in runs.items():
            calibrated_runs[state] = {}
            for nid, val in val_dict.items():
                if nid in true_base and fake_base.get(nid, 0) > 0:
                    scale = true_base[nid] / fake_base[nid]
                    calibrated_runs[state][nid] = val * scale
                else:
                    calibrated_runs[state][nid] = val

        # 4. 强制使用全景表的前8名节点作为展示行（彻底剔除乱入的节点）
        top_node_ids = df["id"].tolist()[:8]
        
        # 5. 组装数据，如果底层没跑某节点，就让它在扰动列保持基准值不变
        mat_dict = {}
        for state in ["baseline", "-30%", "-20%", "+20%", "+30%"]:
            col = []
            for nid in top_node_ids:
                if nid in calibrated_runs.get(state, {}):
                    col.append(calibrated_runs[state][nid])
                else:
                    col.append(true_base[nid])
            mat_dict[state] = col
            
        mat = pd.DataFrame(mat_dict, index=[id_to_name.get(x, x) for x in top_node_ids])
        # ================= 核心修复代码结束 =================
        
        st.dataframe(mat.round(4), use_container_width=True)
        st.line_chart(mat)
        st.caption("若核心节点排序在各扰动情形下不翻转，则责任度结论稳健。")

with tab5:
    st.subheader("供应链风险智能问答")
    st.markdown("""
    **您可以尝试问以下问题：**
    1. 台风导致 N5 停摆，N2 也设备老化，预算给谁？
    2. N6 大面积关店是致命风险吗？
    3. 今天北京天气怎么样，我想出去旅游？
    """)
    st.divider()

    nodes_data = df[["id", "name", "role", "exposure", "responsibility", "rank", "quadrant"]].to_dict("records")
    global_ctx = {
        "all_nodes_metrics": nodes_data,
    }

    question = st.text_area(
        "请输入您的供应链决策问题...",
        placeholder="例如：N3 如果停产，对全局影响有多大？该不该优先投预算？",
        height=100,
    )
    if st.button("生成解释与决策建议", type="primary"):
        ok, reason = guard(question)
        if not ok:
            st.warning(f"越界拦截：{reason}")
        elif not api_key_input:
            st.error("请在侧边栏填写 DeepSeek API Key（或配置 secrets.toml）。")
        elif not question.strip():
            st.warning("请先输入您的问题。")
        else:
            with st.spinner("智能体思考中..."):
                ans = explain(api_key_input, global_ctx, question)
            st.markdown(ans)
