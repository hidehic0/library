from typing import List


class PotentialUnionFind:
    def __init__(self, n: int) -> None:
        """
        重み付きunionfind
        俗に言う、牛ゲー

        uniteは、差を指定して、uniteします
        """
        self.data: List[int] = [-1] * n
        self.pot: List[int] = [0] * n

    def root(self, vtx: int) -> int:
        """
        頂点vtxの親を出力します
        ポテンシャルは出力しません
        """
        if self.data[vtx] < 0:
            return vtx

        rt = self.root(self.data[vtx])
        self.pot[vtx] += self.pot[self.data[vtx]]
        self.data[vtx] = rt

        return rt

    def potential(self, vtx: int) -> int:
        """
        頂点vtxのポテンシャルを出力します
        """
        self.root(vtx)

        return self.pot[vtx]

    def same(self, a: int, b: int) -> bool:
        """
        頂点aと頂点bが同じ連結成分かを判定します
        """
        return self.root(a) == self.root(b)

    def unite(self, a: int, b: int, p: int) -> bool:
        """
        頂点aから頂点bを、pの距離でmergeします
        計算量はlog nです
        """
        p += self.potential(b) - self.potential(a)
        a, b = self.root(a), self.root(b)

        if a == b:
            return False

        if self.data[a] < self.data[b]:
            a, b = b, a
            p *= -1  # ポテンシャルもswapします

        self.data[b] += self.data[a]
        self.data[a] = b
        self.pot[a] = p

        return True

    def diff(self, a: int, b: int) -> int:
        """
        頂点aから頂点bの距離を、出力します
        """

        return self.potential(a) - self.potential(b)
