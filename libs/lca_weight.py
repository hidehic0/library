from collections import defaultdict
import math


class WeightedTreeLCA:
    def __init__(self, n):
        """初期化: ノード数nの木を構築（0-indexed）"""
        self.n = n
        self.log = math.ceil(math.log2(n)) + 1
        self.adj = defaultdict(list)  # 隣接リスト: {ノード: [(隣接ノード, 重み), ...]}
        self.depth = [0] * n  # 各ノードの深さ
        self.dist = [0] * n  # 根からの重み合計
        self.ancestor = [[-1] * self.log for _ in range(n)]  # ダブリングテーブル

    def add_edge(self, u, v, w):
        """辺を追加: uとvを重みwで接続"""
        self.adj[u].append((v, w))
        self.adj[v].append((u, w))

    def dfs(self, u, parent, d, w):
        """DFSで深さ、距離、親を計算"""
        self.depth[u] = d
        self.dist[u] = w
        for v, weight in self.adj[u]:
            if v != parent:
                self.ancestor[v][0] = u
                self.dfs(v, u, d + 1, w + weight)

    def build(self, root=0):
        """ダブリングテーブルの構築"""
        # DFSで初期情報収集
        self.dfs(root, -1, 0, 0)
        # ダブリングテーブルを埋める
        for k in range(1, self.log):
            for u in range(self.n):
                if self.ancestor[u][k - 1] != -1:
                    self.ancestor[u][k] = self.ancestor[self.ancestor[u][k - 1]][k - 1]

    def lca(self, u, v):
        """ノードuとvのLCAを求める"""
        # 深さを揃える
        if self.depth[u] < self.depth[v]:
            u, v = v, u
        for k in range(self.log - 1, -1, -1):
            if (
                self.ancestor[u][k] != -1
                and self.depth[self.ancestor[u][k]] >= self.depth[v]
            ):
                u = self.ancestor[u][k]
        if u == v:
            return u
        # 同時にジャンプ
        for k in range(self.log - 1, -1, -1):
            if self.ancestor[u][k] != self.ancestor[v][k]:
                u = self.ancestor[u][k]
                v = self.ancestor[v][k]
        return self.ancestor[u][0]

    def get_distance(self, u, v):
        """ノードuとvの間の距離（重みの合計）を求める"""
        lca_node = self.lca(u, v)
        return self.dist[u] + self.dist[v] - 2 * self.dist[lca_node]
