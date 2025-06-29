from typing import Any, Callable


class DualSegmentTree:
    def __init__(self, op: Callable[[Any, Any], Any], e: Any, n: int) -> None:
        """
        区間作用/一点取得のセグメント木
        opは区間作用用の関数
        eは初期値
        vは長さ
        """
        self._op: Callable[[Any, Any], Any] = op
        self._e: Any = e
        self._n: int = n
        self.n: int = 1 << (n - 1).bit_length()
        self.data = [e] * (self.n * 2)

    def apply(self, l, r, x) -> None:
        """
        区間[l,r)にxを適用
        """
        assert 0 <= l <= r <= self.n
        l += self.n
        r += self.n

        while l < r:
            if l & 1:
                self.data[l] = self._op(self.data[l], x)
                l += 1

            if r & 1:
                self.data[r - 1] = self._op(self.data[r - 1], x)

            l >>= 1
            r >>= 1

    def get(self, p: int) -> Any:
        """
        pの値を取得する
        """
        assert 0 <= p < self.n

        res = self._e
        p += self.n

        while p:
            res = self._op(res, self.data[p])
            p >>= 1

        return res
