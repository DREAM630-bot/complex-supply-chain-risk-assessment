"""配置驱动通用有向图博弈归因引擎。
严禁大模型参与任何数值计算：最大流 / 最小割 / Shapley 全部在此模块完成。
"""
import json
import random
import itertools
import networkx as nx

INF = 1e9


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------- 图构建
def build_flow_graph(cfg: dict, disabled=(), cap_scale=None) -> nx.DiGraph:
    """拆点建图：节点容量 -> (n__in -> n__out)，边容量 -> (u__out -> v__in)。"""
    cap_scale = cap_scale or {}
    dis = set(disabled)
    G = nx.DiGraph()

    for n in cfg["nodes"]:
        if n["id"] in dis:
            continue
        cap = float(n["capacity"]) * float(cap_scale.get(n["id"], 1.0))
        G.add_edge(f"{n['id']}__in", f"{n['id']}__out",
                   capacity=cap, kind="node", owner=n["id"])

    for e in cfg["edges"]:
        if e["src"] in dis or e["dst"] in dis:
            continue
        G.add_edge(f"{e['src']}__out", f"{e['dst']}__in",
                   capacity=float(e["capacity"]), kind="edge", owner=e["id"],
                   mode=e.get("mode", "road"))

    G.add_node("__S")
    G.add_node("__T")
    for s in cfg["sources"]:
        if s in dis:
            continue
        G.add_edge("__S", f"{s}__in", capacity=INF, kind="supersrc", owner=f"S->{s}")
    for t in cfg["demand_weight"]:
        if t in dis:
            continue
        G.add_edge(f"{t}__out", "__T", capacity=INF, kind="supersink", owner=f"{t}->T")

    return G


# ---------------------------------------------------------------- 状态评估
def eval_state(cfg: dict, disabled=(), cap_scale=None) -> dict:
    """返回 {weighted: 加权需求满足量, reach: 可达需求权重, fragility: 割集脆弱性}"""
    G = build_flow_graph(cfg, disabled, cap_scale)
    if "__S" not in G or "__T" not in G or not nx.has_path(G, "__S", "__T"):
        return {"weighted": 0.0, "reach": 0.0, "fragility": 1.0, "total": 0.0}

    try:
        total, fd = nx.maximum_flow(G, "__S", "__T")
    except (nx.NetworkXError, nx.NetworkXUnbounded):
        return {"weighted": 0.0, "reach": 0.0, "fragility": 1.0, "total": 0.0}

    weighted, reach = 0.0, 0.0
    for t, w in cfg["demand_weight"].items():
        key = f"{t}__out"
        flow = fd.get(key, {}).get("__T", 0.0)
        weighted += flow * float(w)
        if flow > 1e-6:
            reach += float(w)

    # 最小割脆弱性：割集越小，网络越脆
    try:
        _, (side_s, side_t) = nx.minimum_cut(G, "__S", "__T")
        cross = sum(1 for u in side_s for v in G[u] if v in side_t)
        fragility = 1.0 / (1.0 + cross)
    except Exception:
        fragility = 1.0

    return {"weighted": weighted, "reach": reach, "fragility": fragility, "total": total}


# ---------------------------------------------------------------- 特征函数
def value_function(cfg: dict, base: dict, disabled) -> float:
    """失效集合 S 导致的全局供应保障破坏度 ∈ [0,1]。"""
    st = eval_state(cfg, disabled)
    p = cfg.get("penalty", {"flow": 0.65, "reach": 0.20, "cut": 0.15})

    flow_loss = 0.0
    if base["weighted"] > 1e-9:
        flow_loss = (base["weighted"] - st["weighted"]) / base["weighted"]
    reach_loss = 1.0 - st["reach"]
    frag_gain = max(0.0, st["fragility"] - base["fragility"])

    v = p["flow"] * flow_loss + p["reach"] * reach_loss + p["cut"] * frag_gain
    return float(min(max(v, 0.0), 1.0))


# ---------------------------------------------------------------- Shapley
def shapley_exact(cfg, players, base) -> dict:
    """小图精确枚举（n ≤ 10）。"""
    n = len(players)
    cache = {frozenset(): 0.0}

    def v(S):
        key = frozenset(S)
        if key not in cache:
            cache[key] = value_function(cfg, base, key)
        return cache[key]

    phi = {p: 0.0 for p in players}
    fact = [1] * (n + 1)
    for i in range(1, n + 1):
        fact[i] = fact[i - 1] * i

    for p in players:
        others = [x for x in players if x != p]
        for k in range(len(others) + 1):
            for comb in itertools.combinations(others, k):
                w = fact[k] * fact[n - k - 1] / fact[n]
                phi[p] += w * (v(set(comb) | {p}) - v(set(comb)))
    return phi


def shapley_mc(cfg, players, base, n_perm=300, seed=42) -> dict:
    """蒙特卡罗排列采样 Shapley。"""
    rng = random.Random(seed)
    cache = {frozenset(): 0.0}

    def v(S):
        key = frozenset(S)
        if key not in cache:
            cache[key] = value_function(cfg, base, key)
        return cache[key]

    phi = {p: 0.0 for p in players}
    for _ in range(n_perm):
        perm = players[:]
        rng.shuffle(perm)
        S, prev = set(), v(set())
        for p in perm:
            S.add(p)
            cur = v(S)
            phi[p] += cur - prev
            prev = cur
    return {p: val / n_perm for p, val in phi.items()}


def shapley_hierarchical(cfg, base, n_perm=300, seed=42, top_k=6) -> dict:
    """分层 Shapley：关键候选集算 Shapley，其余聚合为背景块。"""
    nodes = [n["id"] for n in cfg["nodes"]]
    G = build_flow_graph(cfg)

    # 候选集：割点 / 桥 / 高吞吐 / 唯一转换点 / 高入出度
    und = G.to_undirected()
    arts = set(nx.articulation_points(und)) if und.number_of_nodes() > 2 else set()
    bridges = {u for e in nx.bridges(und) for u in e} if und.number_of_nodes() > 2 else set()
    cap_rank = sorted(cfg["nodes"], key=lambda n: -n["capacity"])
    cand = set(arts) | set(bridges) | {n["id"] for n in cap_rank[:top_k]}
    cand |= {n["id"] for n in cfg["nodes"] if n.get("critical")}
    cand = {c for c in cand if c in nodes}
    if not cand:
        cand = {n["id"] for n in cap_rank[:top_k]}

    background = [n for n in nodes if n not in cand]
    players = sorted(cand) + (["__BACKGROUND__"] if background else [])

    def v_block(S):
        real = set()
        for s in S:
            if s == "__BACKGROUND__":
                real |= set(background)
            else:
                real.add(s)
        return value_function(cfg, base, real)

    rng = random.Random(seed)
    cache = {frozenset(): 0.0}

    def v(S):
        key = frozenset(S)
        if key not in cache:
            cache[key] = v_block(key)
        return cache[key]

    phi = {p: 0.0 for p in players}
    for _ in range(n_perm):
        perm = players[:]
        rng.shuffle(perm)
        S, prev = set(), v(set())
        for p in perm:
            S.add(p)
            cur = v(S)
            phi[p] += cur - prev
            prev = cur
    phi = {p: val / n_perm for p, val in phi.items()}

    # 背景值均摊回未入选节点
    if "__BACKGROUND__" in phi and background:
        share = phi.pop("__BACKGROUND__") / len(background)
        for b in background:
            phi[b] = share
    return phi


def normalize_phi(phi: dict) -> dict:
    s = sum(abs(v) for v in phi.values())
    if s < 1e-12:
        return {k: 0.0 for k in phi}
    return {k: v / s for k, v in phi.items()}


# ---------------------------------------------------------------- 敏感性
def sensitivity_scan(cfg, nodes, n_perm=120, deltas=(-0.3, -0.2, 0.2, 0.3), seed=42):
    """对代理/先验节点做容量扰动，返回各情形下责任度向量。"""
    base = eval_state(cfg)
    base_phi = normalize_phi(shapley_mc(cfg, nodes, base, n_perm, seed))

    targets = [n["id"] for n in cfg["nodes"]
               if n.get("capacity_source") in ("proxy_ratio", "prior_assumption")]
    runs = {"baseline": base_phi}
    for d in deltas:
        scale = {t: 1.0 + d for t in targets}
        st = eval_state(cfg, (), scale)
        runs[f"{d:+.0%}"] = normalize_phi(shapley_mc(cfg, nodes, st, n_perm, seed))
    return runs