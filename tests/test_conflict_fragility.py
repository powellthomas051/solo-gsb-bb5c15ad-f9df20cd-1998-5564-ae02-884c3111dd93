"""conflict-fragility-map 接口端到端测试与 Tarjan 割点/桥单测。"""

from __future__ import annotations

import os
import random

import pytest

from app.fragility import analyze_conflict_fragility


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp("fragility_db")
    os.environ["PRIMER_DB_PATH"] = str(tmp_dir / "test.db")

    from app import db, main

    db.init_db()
    from fastapi.testclient import TestClient

    with TestClient(main.app) as c:
        yield c


# 各引物的风险画像（阈值 ct=4/tt=3，已用 analyze_pair 独立核算）：
#   异二聚体 high 边：P0-P1、P1-P2、P2-P3（路径）、P3-P4、P3-P5、P4-P5（三角形）
#   P6 自二聚体 high（与其余引物均 low，孤立顶点）；P7 全部 low（孤立顶点）
#   其余异二聚体对全部 low，无 medium
PAYLOAD = {
    "primers": [
        {"name": "P0", "sequence": "TCGCTGACGG"},
        {"name": "P1", "sequence": "CCCGTCACCC"},
        {"name": "P2", "sequence": "GTAGGGGATG"},
        {"name": "P3", "sequence": "CCTCATTCAT"},
        {"name": "P4", "sequence": "AACTCGATGC"},
        {"name": "P5", "sequence": "TAGGCGCAGG"},
        {"name": "P6", "sequence": "AATTGCTCAA"},
        {"name": "P7", "sequence": "AAACGGAGAT"},
    ],
    "continuous_threshold": 4,
    "three_prime_threshold": 3,
}


@pytest.fixture(scope="module")
def batch_id(client):
    return client.post("/api/batches", json=PAYLOAD).json()["id"]


def _map(client, batch_id, levels=("high",)):
    return client.post(
        f"/api/batches/{batch_id}/conflict-fragility-map",
        json={"risk_levels": list(levels)},
    )


def _positions(members):
    return [m["position"] for m in members]


# ---------- 端到端：割点、桥、固有自风险 ----------

def test_fragility_map_full_structure(client, batch_id):
    resp = _map(client, batch_id)
    assert resp.status_code == 200, resp.text
    data = resp.json()

    assert data["batch_id"] == batch_id
    assert data["risk_levels"] == ["high"]
    assert data["primer_count"] == 8
    assert data["conflict_edge_count"] == 6
    # 分量：{P0..P5}、{P6}、{P7}（孤立引物各算一个）
    assert data["component_count"] == 3
    assert data["articulation_point_count"] == 3
    assert data["bridge_count"] == 3

    # 割点按批次位置升序：P1、P2、P3
    aps = data["articulation_points"]
    assert [a["position"] for a in aps] == [1, 2, 3]
    assert [a["name"] for a in aps] == ["P1", "P2", "P3"]

    # P1：删除后分裂为 {P0} 与 {P2,P3,P4,P5}；全图分量数 3-1+2=4
    assert aps[0]["components_after_removal"] == 4
    assert [_positions(g) for g in aps[0]["split_groups"]] == [[0], [2, 3, 4, 5]]
    # P2：{P0,P1} 与 {P3,P4,P5}
    assert aps[1]["components_after_removal"] == 4
    assert [_positions(g) for g in aps[1]["split_groups"]] == [[0, 1], [3, 4, 5]]
    # P3：{P0,P1,P2} 与 {P4,P5}
    assert aps[2]["components_after_removal"] == 4
    assert [_positions(g) for g in aps[2]["split_groups"]] == [[0, 1, 2], [4, 5]]

    # 分裂组成员带名称且与位置一致
    member = aps[0]["split_groups"][0][0]
    assert member == {"position": 0, "name": "P0"}

    # 桥按两端位置升序：P0-P1、P1-P2、P2-P3（三角形边不是桥）
    bridges = data["bridges"]
    assert [(b["position_a"], b["position_b"]) for b in bridges] == [
        (0, 1),
        (1, 2),
        (2, 3),
    ]
    assert [(b["name_a"], b["name_b"]) for b in bridges] == [
        ("P0", "P1"),
        ("P1", "P2"),
        ("P2", "P3"),
    ]
    assert all(b["risk_level"] == "high" for b in bridges)
    # 桥两侧成员（side_a 为位置较小端一侧）
    assert _positions(bridges[0]["side_a"]) == [0]
    assert _positions(bridges[0]["side_b"]) == [1, 2, 3, 4, 5]
    assert _positions(bridges[1]["side_a"]) == [0, 1]
    assert _positions(bridges[1]["side_b"]) == [2, 3, 4, 5]
    assert _positions(bridges[2]["side_a"]) == [0, 1, 2]
    assert _positions(bridges[2]["side_b"]) == [3, 4, 5]

    # 自二聚体不参与连通性，但作为固有自风险返回
    assert data["intrinsic_self_risks"] == [
        {"position": 6, "name": "P6", "risk_level": "high"}
    ]


def test_risk_levels_normalized_and_deduped(client, batch_id):
    resp = _map(client, batch_id, levels=("HIGH", " high ", "high"))
    assert resp.status_code == 200, resp.text
    assert resp.json()["risk_levels"] == ["high"]


def test_medium_level_selects_empty_graph(client, batch_id):
    # 批次内无 medium 配对：图为空，全部引物为孤立顶点
    resp = _map(client, batch_id, levels=("medium",))
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["conflict_edge_count"] == 0
    assert data["component_count"] == 8
    assert data["articulation_points"] == []
    assert data["bridges"] == []
    assert data["intrinsic_self_risks"] == []


def test_deterministic_repeated_calls(client, batch_id):
    r1 = _map(client, batch_id).json()
    r2 = _map(client, batch_id).json()
    assert r1 == r2


def test_batch_not_modified(client, batch_id):
    before = client.get(f"/api/batches/{batch_id}").json()
    resp = _map(client, batch_id, levels=("high", "medium", "low"))
    assert resp.status_code == 200
    after = client.get(f"/api/batches/{batch_id}").json()
    assert after == before


# ---------- 错误处理 ----------

def test_batch_not_found_404(client):
    resp = _map(client, "no-such-batch")
    assert resp.status_code == 404


def test_invalid_risk_level_422(client, batch_id):
    resp = _map(client, batch_id, levels=("critical",))
    assert resp.status_code == 422


def test_empty_risk_levels_422(client, batch_id):
    resp = _map(client, batch_id, levels=())
    assert resp.status_code == 422


# ---------- Tarjan 割点/桥单测 ----------

def test_empty_graph_all_isolated():
    result = analyze_conflict_fragility(3, [])
    assert result.component_count == 3
    assert result.edge_count == 0
    assert result.articulation_points == ()
    assert result.bridges == ()


def test_single_edge_is_bridge_no_cut():
    result = analyze_conflict_fragility(2, [(0, 1)])
    assert result.component_count == 1
    assert result.articulation_points == ()
    assert len(result.bridges) == 1
    br = result.bridges[0]
    assert (br.position_a, br.position_b) == (0, 1)
    assert br.side_a == (0,)
    assert br.side_b == (1,)


def test_path_middle_is_cut():
    # 路径 0-1-2：1 是割点，分裂为 {0} 与 {2}；两条边都是桥
    result = analyze_conflict_fragility(3, [(0, 1), (1, 2)])
    assert result.component_count == 1
    assert len(result.articulation_points) == 1
    ap = result.articulation_points[0]
    assert ap.position == 1
    assert ap.components_after_removal == 2
    assert ap.split_groups == ((0,), (2,))
    assert [(b.position_a, b.position_b) for b in result.bridges] == [(0, 1), (1, 2)]


def test_triangle_no_cut_no_bridge():
    result = analyze_conflict_fragility(3, [(0, 1), (1, 2), (0, 2)])
    assert result.component_count == 1
    assert result.articulation_points == ()
    assert result.bridges == ()


def test_star_center_cut_with_singleton_groups():
    # 星型 0-1,0-2,0-3：中心 0 为割点，每个叶子各成一个分裂组
    result = analyze_conflict_fragility(4, [(0, 1), (0, 2), (0, 3)])
    assert len(result.articulation_points) == 1
    ap = result.articulation_points[0]
    assert ap.position == 0
    assert ap.components_after_removal == 3
    assert ap.split_groups == ((1,), (2,), (3,))
    assert len(result.bridges) == 3


def test_parallel_edges_deduped_and_bridge_detected():
    # 平行重复边去重为单边：去重后 0-1 是桥（若不去重会被误判为非桥）
    result = analyze_conflict_fragility(2, [(0, 1), (1, 0), (0, 1)])
    assert result.edge_count == 1
    assert len(result.bridges) == 1


def test_self_loop_ignored_for_connectivity():
    # 自环（自二聚体）不参与连通性：0 仍与 1 仅由单边相连
    result = analyze_conflict_fragility(3, [(0, 0), (0, 1), (2, 2)])
    assert result.component_count == 2
    assert result.edge_count == 1
    assert result.articulation_points == ()
    assert [(b.position_a, b.position_b) for b in result.bridges] == [(0, 1)]


def test_disconnected_graph_components_and_sides():
    # 分量一：路径 0-1-2；分量二：单边 3-4；顶点 5 孤立
    result = analyze_conflict_fragility(6, [(0, 1), (1, 2), (3, 4)])
    assert result.component_count == 3
    assert len(result.articulation_points) == 1
    ap = result.articulation_points[0]
    assert ap.position == 1
    # 删除顶点 1 后全图分量数：3 - 1 + 2 = 4
    assert ap.components_after_removal == 4
    assert ap.split_groups == ((0,), (2,))
    # 桥按两端位置升序，跨越各分量
    assert [(b.position_a, b.position_b) for b in result.bridges] == [
        (0, 1),
        (1, 2),
        (3, 4),
    ]
    last = result.bridges[-1]
    assert last.side_a == (3,)
    assert last.side_b == (4,)


def test_bridge_sides_follow_position_order_not_dfs_order():
    # 路径 2-0-1（边 (2,0)、(0,1)）：桥 (0,2) 的 position_a=0 一侧为 {0,1}
    result = analyze_conflict_fragility(3, [(2, 0), (0, 1)])
    assert [(b.position_a, b.position_b) for b in result.bridges] == [(0, 1), (0, 2)]
    first, second = result.bridges
    assert first.side_a == (0, 2)
    assert first.side_b == (1,)
    assert second.side_a == (0, 1)
    assert second.side_b == (2,)


# ---------- 与暴力定义对照的随机化测试 ----------

def _brute_fragility(n, edges):
    """按定义暴力求解：逐点/逐边删除后数连通分量（BFS）。"""
    es = set()
    for u, v in edges:
        if u == v:
            continue
        es.add((min(u, v), max(u, v)))
    adj = [set() for _ in range(n)]
    for a, b in es:
        adj[a].add(b)
        adj[b].add(a)

    def components(exclude_v=None, exclude_edge=None):
        seen = [False] * n
        groups = []
        for s in range(n):
            if s == exclude_v or seen[s]:
                continue
            seen[s] = True
            stack = [s]
            comp = []
            while stack:
                x = stack.pop()
                comp.append(x)
                for y in adj[x]:
                    if y == exclude_v or seen[y]:
                        continue
                    if exclude_edge is not None and {x, y} == set(exclude_edge):
                        continue
                    seen[y] = True
                    stack.append(y)
            groups.append(tuple(sorted(comp)))
        return groups

    base = components()
    base_count = len(base)

    aps = {}
    for v in range(n):
        after = components(exclude_v=v)
        if len(after) > base_count:
            vcomp = next(g for g in base if v in g)
            split = tuple(
                sorted(
                    (g for g in after if set(g) < set(vcomp)),
                    key=lambda g: g[0],
                )
            )
            aps[v] = (len(after), split)

    bridges = {}
    for a, b in sorted(es):
        after = components(exclude_edge=(a, b))
        if len(after) > base_count:
            side_a = next(g for g in after if a in g)
            side_b = next(g for g in after if b in g)
            bridges[(a, b)] = (side_a, side_b)
    return base_count, len(es), aps, bridges


def test_matches_brute_force_randomized():
    rng = random.Random(20260920)
    for _ in range(3000):
        n = rng.randint(1, 9)
        edges = []
        for u in range(n):
            for v in range(u, n):
                if rng.random() < 0.3:
                    edges.append((u, v))
                    # 混入平行重复边与自环，验证去重与忽略语义
                    if rng.random() < 0.3:
                        edges.append((v, u))
        result = analyze_conflict_fragility(n, edges)
        base_count, edge_count, aps, bridges = _brute_fragility(n, edges)

        assert result.component_count == base_count
        assert result.edge_count == edge_count

        assert [ap.position for ap in result.articulation_points] == sorted(aps)
        for ap in result.articulation_points:
            count_after, split = aps[ap.position]
            assert ap.components_after_removal == count_after
            assert ap.split_groups == split

        assert [(b.position_a, b.position_b) for b in result.bridges] == sorted(
            bridges
        )
        for b in result.bridges:
            side_a, side_b = bridges[(b.position_a, b.position_b)]
            assert b.side_a == side_a
            assert b.side_b == side_b
