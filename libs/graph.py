# グラフ構造
# 無向グラフ
from collections import deque
from typing import List
from typing_extensions import Tuple


class Graph:
    """
    グラフ構造体
    """

    def __init__(self, N: int, dire: bool = False) -> None:
        """
        Nは頂点数、direは有向グラフかです
        """
        self.N = N
        self.dire = dire
        self.grath = [[] for _ in [0] * self.N]
        self.in_deg = [0] * N

    def new_side(self, a: int, b: int):
        """
        注意　0-indexedが前提
        aとbを辺で繋ぎます
        有向グラフなら、aからbだけ、無向グラフなら、aからbと、bからaを繋ぎます
        """
        self.grath[a].append(b)
        if self.dire:
            self.in_deg[b] += 1

        if not self.dire:
            self.grath[b].append(a)

    def side_input(self):
        """
        標準入力で、新しい辺を追加します
        """
        a, b = map(lambda x: int(x) - 1, input().split())
        self.new_side(a, b)

    def input(self, M: int):
        """
        標準入力で複数行受け取り、各行の内容で辺を繋ぎます
        """
        for _ in [0] * M:
            self.side_input()

    def get(self, a: int):
        """
        頂点aの隣接頂点を出力します
        """
        return self.grath[a]

    def all(self) -> List[List[int]]:
        """
        グラフの隣接リストをすべて出力します
        """
        return self.grath

    def topological(self, unique: bool = False) -> List[int]:
        """
        トポロジカルソートします
        有向グラフ限定です

        引数のuniqueは、トポロジカルソート結果が、一意に定まらないとエラーを吐きます
        閉路がある、または、uniqueがTrueで一意に定まらなかった時は、[-1]を返します
        """
        if not self.dire:
            raise ValueError("グラフが有向グラフでは有りません (╥﹏╥)")

        in_deg = self.in_deg[:]

        S: deque[int] = deque([])
        order: List[int] = []

        for i in range(self.N):
            if in_deg[i] == 0:
                S.append(i)

        while S:
            if unique and len(S) != 1:
                return [-1]

            cur = S.pop()
            order.append(cur)

            for nxt in self.get(cur):
                in_deg[nxt] -= 1

                if in_deg[nxt] == 0:
                    S.append(nxt)

        if sum(in_deg) > 0:
            return [-1]
        else:
            return [x for x in order]


class GraphW:
    """
    重み付きグラフ
    """

    def __init__(self, N: int, dire: bool = False) -> None:
        self.N = N
        self.dire = dire
        self.grath = [[] for _ in [0] * self.N]

    def new_side(self, a: int, b: int, w: int):
        """
        注意　0-indexedが前提
        aとbを辺で繋ぎます
        有向グラフなら、aからbだけ、無向グラフなら、aからbと、bからaを繋ぎます
        """
        self.grath[a].append((b, w))
        if not self.dire:
            self.grath[b].append((a, w))

    def side_input(self):
        """
        標準入力で、新しい辺を追加します
        """
        a, b, w = map(lambda x: int(x) - 1, input().split())
        self.new_side(a, b, w + 1)

    def input(self, M: int):
        """
        標準入力で複数行受け取り、各行の内容で辺を繋ぎます
        """
        for _ in [0] * M:
            self.side_input()

    def get(self, a: int) -> List[Tuple[int]]:
        """
        頂点aの隣接頂点を出力します
        """
        return self.grath[a]

    def all(self) -> List[List[Tuple[int]]]:
        """
        グラフの隣接リストをすべて出力します
        """
        return self.grath
