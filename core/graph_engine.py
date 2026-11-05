"""配置驱动通用有向图博弈归因引擎。"""
import json
import random
import itertools
import networkx as nx

INF = 1e9

def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_flow_graph(cfg: dict, disabled=(), cap_scale=None) -> nx.DiGraph:
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

def eval_state(cfg: dict, disabled=(), cap_scale=None) -> dict:
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

    # 割集脆弱性：用最小割容量与当前最大流的比值衡量
    try:
        min_cut_val, (side_s, side_t) = nx.minimum_cut(G, "__S", "__T")
        fragility = 1.0 / (1.0 + min_cut_val / max(total, 1e-9))
    except Exception:
        fragility = 1.0

    return {"weighted": weighted, "reach": reach, "fragility": fragility, "total": total}

def value_function(cfg: dict, base: dict, disabled) -> float:
    st = eval_state(cfg, disabled)
    p = cfg.get("penalty", {"flow": 0.65, "reach": 0.20, "cut": 0.15})

    flow_loss = 0.0
    if base["weighted"] > 1e-9:
        flow_loss = (base["weighted"] - st["weighted"]) / base["weighted"]
    
    # 修复：基于基准状态的比例
    reach_loss = max(0.0, (base["reach"] - st["reach"]) / max(base["reach"], 1e-9))
    frag_gain = max(0.0, st["fragility"] - base["fragility"])

    v = p["flow"] * flow_loss + p["reach"] * reach_loss + p["cut"] * frag_gain
    return float(min(max(v, 0.0), 1.0))

def shapley_mc(cfg, players, base, n_perm=300, seed=42) -> dict:
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
    nodes = [n["id"] for n in cfg["nodes"]]
    G = build_flow_graph(cfg)

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

def sensitivity_scan(cfg, nodes, n_perm=120, deltas=(-0.3, -0.2, 0.2, 0.3), seed=42):
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
