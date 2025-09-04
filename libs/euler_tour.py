from typing import List, Tuple


class EulerTour:
    def __init__(self, edges: List[Tuple[int, int, int]], root: int = 0) -> None:
        """
        edges[i] = (u, v, w)
        なお閉路がない、連結という前提 エラー処理をしていない

        木上の最短経路は、path_query関数を使うこと

        初期化にO(N + M) それ以外は、$O(log n)$

        Warning:
        ac-library-pythonを__init__内で使用しているので注意
        定数倍が遅い事に注意 あとメモリも注意 結構リストを使用している
        """
        # assert len(edges) >= 1

        from atcoder.segtree import SegTree

        self.edges = edges
        self._n = max([max(u, v) for u, v, w in edges]) + 1
        self.root = root
        self.graph: List[List[Tuple[int, int, int]]] = [[] for _ in [0] * self._n]

        for i, (u, v, w) in enumerate(edges):
            self.graph[u].append((v, w, i))
            self.graph[v].append((u, w, i))

        self._build()

        self.segtree_edgecost = SegTree(lambda a, b: a + b, 0, self.edge_cost)
        self.segtree_depth = SegTree(
            min,
            (1 << 63, 1 << 63),
            [(d, i) for i, d in enumerate(self.depth)],
        )

        return

    def _build(self) -> None:
        self.euler_tour: List[Tuple[int, int]] = [(0, -1)]
        self.edge_cost: List[int] = [0]
        self.depth: List[int] = [0]

        def dfs(cur: int, p: int = -1, d: int = 0) -> None:
            for nxt, w, i in self.graph[cur]:
                if nxt == p:
                    continue

                self.euler_tour.append((nxt, i))
                self.edge_cost.append(w)
                self.depth.append(d + 1)
                dfs(nxt, cur, d + 1)
                self.euler_tour.append((cur, i))
                self.edge_cost.append(-w)
                self.depth.append(d)

        dfs(self.root)

        self.first_arrival = [-1] * self._n
        self.last_arrival = [-1] * self._n
        self.first_arrival[self.root] = 0
        self.last_arrival[self.root] = len(self.euler_tour) - 1
        self.edge_plus = [-1] * (self._n - 1)
        self.edge_minus = [-1] * (self._n - 1)

        for i, (u, edge_ind) in enumerate(self.euler_tour):
            if self.edge_cost[i] >= 0:
                self.edge_plus[edge_ind] = i
            else:
                self.edge_minus[edge_ind] = i

            if self.first_arrival[u] == -1:
                self.first_arrival[u] = i

            self.last_arrival[u] = i

    def lca(self, a: int, b: int) -> int:
        # assert 0 <= a < self._n and 0 <= b < self._n

        l, r = (
            min(self.first_arrival[a], self.first_arrival[b]),
            max(self.last_arrival[a], self.last_arrival[b]),
        )

        return self.euler_tour[self.segtree_depth.prod(l, r)[1]][0]

    def path_query_from_root(self, u: int) -> int:
        assert 0 <= u < self._n
        return self.segtree_edgecost.prod(0, self.first_arrival[u] + 1)

    def path_query(self, a: int, b: int) -> int:
        """
        aからbへの最短経路
        """
        # assert 0 <= a < self._n and 0 <= b < self._n
        try:
            l = self.lca(a, b)
        except IndexError:
            return 0

        return (
            self.path_query_from_root(a)
            + self.path_query_from_root(b)
            - (2 * self.path_query_from_root(l))
        )

    def change_edge_cost(self, i: int, w: int) -> None:
        # assert 0 <= i < len(self.edges)
        self.segtree_edgecost.set(self.edge_plus[i], w)
        self.segtree_edgecost.set(self.edge_minus[i], -w)
