from collections import defaultdict


# UnionFind木
class UnionFind:
    def __init__(self, n: int) -> None:
        """UnionFind

        rollbackをデフォルトで装備済み
        計算量は経路圧縮を行わないため、基本的なUnionFindの動作は、一回あたり、O(log N)
        rollbackは、一回あたり、O(1)で行える。
        """
        self.size = n
        self.data = [-1] * n
        self.hist = []

    def leader(self, vtx: int) -> int:
        """頂点vtxの親を出力します"""
        if self.data[vtx] < 0:
            return vtx

        return self.leader(self.data[vtx])

    def same(self, a: int, b: int) -> bool:
        """aとbが連結しているかどうか判定します"""
        return self.leader(a) == self.leader(b)

    def merge(self, a: int, b: int) -> bool:
        """aとbを結合します

        leaderが同じでも、履歴には追加します
        """
        ra, rb = self.leader(a), self.leader(b)

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

    def rollback(self) -> bool:
        """Undo

        redoはありません
        """
        if not self.hist:
            return False

        ra, rb, da, db = self.hist.pop()
        self.data[ra] = da
        self.data[rb] = db
        return True

    def all(self) -> list[list[int]]:
        D = defaultdict(list)

        for i in range(self.size):
            D[self.leader(i)].append(i)

        return [l for l in D.values()]
