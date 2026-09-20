"""异二聚体冲突图的脆弱性分析：Tarjan low-link 一次遍历求全部割点与桥。

冲突图
------
顶点为批次内全部引物（顶点号即批次位置；无冲突边的引物是孤立顶点，仍计
入连通分量）。命中所选风险等级的**异二聚体**配对在两端引物间连一条无向
边；**自二聚体不进入冲突图**（不参与连通性，由上层作为
intrinsic_self_risks 单独返回）。同一对引物的重复边（平行边）先去重为
单边：平行边若不去重会被误判为“非桥”，去重后每对引物至多一条边，桥与
割点才是标准的图论语义。

Tarjan low-link 一次遍历
------------------------
图可不连通：按顶点编号升序对每个未访问分量做**迭代式** DFS（邻点按编号
升序访问，结果确定可复现），同一遍遍历中得到每个顶点的发现时间 tin、
经回边可达的最早发现时间 low、DFS 父指针与欧拉区间 [tin, tout]：

* 非根顶点 v 是割点 ⟺ 存在 DFS 子节点 u 使 low[u] ≥ tin[v]；
  DFS 根是割点 ⟺ 它在 DFS 树中有 ≥ 2 个子节点；
* 树边 (v, u)（v 为父）是桥 ⟺ low[u] > tin[v]。

无向图 DFS 的非树边都是连向祖先的回边，因此分裂组与桥两侧可直接由
DFS 树结构读出，无需再次遍历：

* 割点 v 的每个满足 low[u] ≥ tin[v] 的子树在删除 v 后恰好成为一个分裂组；
  v 所在分量中不属于这些子树的其余顶点（非根时即“父侧”，必非空；根时
  为空）构成最后一个分裂组；
* 桥 (v, u) 删除后，u 一侧恰为 u 的子树，另一侧为分量其余顶点。

子树即欧拉区间 ``euler[tin[v]..tout[v]]``，分裂组与桥两侧成员由区间切片
得到。全部成员列表按批次位置升序；同一割点的分裂组按组内最小位置升序；
割点按位置、桥按两端位置升序，结果确定可复现。总复杂度 O(V + E)。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ArticulationPoint:
    """一个割点（顶点号即批次位置）。

    components_after_removal: 从整张冲突图删除该顶点后的连通分量数
        （其余分量不受影响，其所在分量分裂为 split_groups 列出的各组，
        故恒等于 删除前分量数 - 1 + 分裂组数）；
    split_groups: 删除该顶点后其所在连通分量分裂出的各组成员
        （组内按批次位置升序，组间按组内最小位置升序）。
    """

    position: int
    components_after_removal: int
    split_groups: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class Bridge:
    """一座桥（无向边，position_a < position_b）。

    side_a / side_b: 删除该桥后 position_a / position_b 所在一侧的
    连通分量成员（均按批次位置升序）。
    """

    position_a: int
    position_b: int
    side_a: tuple[int, ...]
    side_b: tuple[int, ...]


@dataclass(frozen=True)
class FragilityMap:
    """冲突图脆弱性分析结果。

    component_count: 删除任何顶点之前的连通分量数（孤立顶点各算一个）；
    edge_count: 去重后的无向冲突边数；
    articulation_points: 全部割点（按批次位置升序）；
    bridges: 全部桥（按两端批次位置升序）。
    """

    component_count: int
    edge_count: int
    articulation_points: tuple[ArticulationPoint, ...]
    bridges: tuple[Bridge, ...]


def analyze_conflict_fragility(
    primer_count: int, edges: list[tuple[int, int]]
) -> FragilityMap:
    """在异二聚体冲突图上用 Tarjan low-link 一次遍历求全部割点与桥。

    ``edges`` 元素为 ``(位置u, 位置v)``；允许平行重复边（内部去重）与
    自环（自二聚体，忽略、不参与连通性）。顶点编号为 0..primer_count-1。
    """
    n = primer_count

    # 平行重复边去重；自环（自二聚体）不参与连通性
    unique_edges: set[tuple[int, int]] = set()
    for u, v in edges:
        if u == v:
            continue
        unique_edges.add((u, v) if u < v else (v, u))

    adjacency: list[list[int]] = [[] for _ in range(n)]
    for u, v in unique_edges:
        adjacency[u].append(v)
        adjacency[v].append(u)
    for neighbors in adjacency:
        neighbors.sort()

    tin = [-1] * n          # 发现时间（即欧拉序号）
    low = [0] * n           # 经回边可达的最早发现时间
    tout = [-1] * n         # 子树在欧拉序中的右端点（含）
    parent = [-1] * n       # DFS 父指针（根为 -1）
    comp_root = [-1] * n    # 所在 DFS 树的根（标识连通分量）
    children: list[list[int]] = [[] for _ in range(n)]
    euler: list[int] = []   # euler[tin[v]] == v
    timer = 0
    component_count = 0

    for start in range(n):
        if tin[start] != -1:
            continue
        component_count += 1
        comp_root[start] = start
        tin[start] = low[start] = timer
        euler.append(start)
        timer += 1
        # 迭代式 DFS：栈元素为 (顶点, 下一个待处理邻点下标)
        stack: list[tuple[int, int]] = [(start, 0)]
        while stack:
            v, i = stack[-1]
            if i < len(adjacency[v]):
                u = adjacency[v][i]
                stack[-1] = (v, i + 1)
                if u == parent[v]:
                    continue  # 树边回父（边已去重，父边唯一）
                if tin[u] != -1:
                    # 回边：无向图中已访问邻点必为祖先（或子孙，此时
                    # tin[u] > tin[v] >= low[v]，取 min 无影响）
                    if tin[u] < low[v]:
                        low[v] = tin[u]
                else:
                    parent[u] = v
                    comp_root[u] = comp_root[v]
                    children[v].append(u)
                    tin[u] = low[u] = timer
                    euler.append(u)
                    timer += 1
                    stack.append((u, 0))
            else:
                stack.pop()
                tout[v] = timer - 1
                if stack:
                    p = stack[-1][0]
                    if low[v] < low[p]:
                        low[p] = low[v]

    def subtree_members(v: int) -> tuple[int, ...]:
        """v 的 DFS 子树全部顶点（欧拉区间切片，按批次位置升序）。"""
        return tuple(sorted(euler[tin[v] : tout[v] + 1]))

    def component_members(v: int) -> list[int]:
        root = comp_root[v]
        return euler[tin[root] : tout[root] + 1]

    articulation_points: list[ArticulationPoint] = []
    bridges: list[Bridge] = []

    for v in range(n):
        # 满足 low[u] >= tin[v] 的子树在删除 v 后各自成为一个分裂组；
        # 对 DFS 根该条件恒成立（分量内不存在通向根之前的回边）
        separated = [u for u in children[v] if low[u] >= tin[v]]
        if parent[v] == -1:
            is_cut = len(children[v]) >= 2
        else:
            is_cut = len(separated) >= 1
        if not is_cut:
            continue

        groups: list[tuple[int, ...]] = [
            subtree_members(u) for u in separated
        ]
        if parent[v] != -1:
            # 父侧余部：分量中不属于任何分离子树的其余顶点（必非空）
            intervals = [(tin[u], tout[u]) for u in separated]
            rest = [
                x
                for x in component_members(v)
                if x != v
                and not any(a <= tin[x] <= b for a, b in intervals)
            ]
            groups.append(tuple(sorted(rest)))
        groups.sort(key=lambda g: g[0])
        articulation_points.append(
            ArticulationPoint(
                position=v,
                components_after_removal=component_count - 1 + len(groups),
                split_groups=tuple(groups),
            )
        )

    for v in range(n):
        for u in children[v]:
            if low[u] <= tin[v]:
                continue
            # 树边 (v, u) 是桥：u 侧为其子树，v 侧为分量其余顶点
            sub = set(subtree_members(u))
            rest = tuple(
                sorted(x for x in component_members(v) if x not in sub)
            )
            side_u = tuple(sorted(sub))
            if v < u:
                bridges.append(Bridge(v, u, rest, side_u))
            else:
                bridges.append(Bridge(u, v, side_u, rest))
    bridges.sort(key=lambda b: (b.position_a, b.position_b))

    return FragilityMap(
        component_count=component_count,
        edge_count=len(unique_edges),
        articulation_points=tuple(articulation_points),
        bridges=tuple(bridges),
    )
