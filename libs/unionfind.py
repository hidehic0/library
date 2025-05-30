from collections import defaultdict
from typing import List


# UnionFind木
class UnionFind:
    """
    rollbackをデフォルトで装備済み
    計算量は、経路圧縮を行わないため、基本的なUnionFindの動作は、一回あたり、O(log N)
    rollbackは、一回あたり、O(1)で行える。
    """

    def __init__(self, n: int) -> None:
        self.size = n
        self.data = [-1] * n
        self.hist = []

    def root(self, vtx: int) -> int:
        """
        頂点vtxの親を出力します
        """
        if self.data[vtx] < 0:
            return vtx

        return self.root(self.data[vtx])

    def same(self, a: int, b: int):
        """
        aとbが連結しているかどうか判定します
        """
        return self.root(a) == self.root(b)

    def unite(self, a: int, b: int) -> bool:
        """
        aとbを結合します
        rootが同じでも、履歴には追加します
        """
        ra, rb = self.root(a), self.root(b)

        # 履歴を作成する
        new_hist = [ra, rb, self.data[ra], self.data[rb]]
        self.hist.append(new_hist)

        if ra == rb:
            return False

        if self.data[ra] > self.data[rb]:
            ra, rb = rb, ra

        self.data[ra] += self.data[rb]
        self.data[rb] = ra

        return True

    def rollback(self):
        """
        undoします
        redoはありません
        """
        if not self.hist:
            return False

        ra, rb, da, db = self.hist.pop()
        self.data[ra] = da
        self.data[rb] = db
        return True

    def all(self) -> List[List[int]]:
        D = defaultdict(list)

        for i in range(self.size):
            D[self.root(i)].append(i)

        res = []

        for l in D.values():
            res.append(l)

        return res
