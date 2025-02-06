from typing import List


class BIT:
    """
    BITです
    要素更新と、区間和を求める事ができます
    1-indexedです

    計算量は、一回の動作につきすべてO(log n)です
    """

    def __init__(self, n: int) -> None:
        self.n: int = n
        self.bit: List[int] = [0] * (n + 1)

    def sum(self, i: int) -> int:
        """
        i番目までの和を求めます
        計算量は、O(log n)です
        """
        res = 0

        while i:
            res += self.bit[i]
            i -= -i & i

        return res

    def interval_sum(self, l: int, r: int) -> None:
        """
        lからrまでの総和を求められます
        lは0-indexedで、rは1-indexedにしてください
        """
        return self.sum(r) - self.sum(l)

    def add(self, i: int, x: int):
        """
        i番目の要素にxを足します
        計算量は、O(log n)です
        """
        if i == 0:
            raise IndexError("このデータ構造は、1-indexedです")

        while i <= self.n:
            self.bit[i] += x
            i += -i & i
