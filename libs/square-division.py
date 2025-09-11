import math
from collections.abc import Callable
from typing import Any


class SquareDivision:
    def __init__(self, lis: list[Any], op: Callable[[Any, Any], Any]) -> None:
        """平方分割ライブラリ

        ほぼACLのセグ木と同じ
        """
        self.n = len(lis)
        self.op = op
        self.block_size = math.isqrt(self.n)
        self.blocks = []
        self.lis = lis[:]

        for i in range(0, self.n, self.block_size):
            block_val = lis[i]
            for k in range(i + 1, min(i + self.block_size, self.n)):
                block_val = self.op(block_val, lis[k])
            self.blocks.append(block_val)

        self.m = len(self.blocks)

    def get_block_index_left(self, i: int) -> int:
        return i // self.block_size

    def get_block_index_right(self, i: int) -> int:
        return (i + self.block_size - 1) // self.block_size

    def prod(self, l: int, r: int) -> Any:
        """rは0-indexedなのに注意してください"""
        assert 0 <= l <= r < self.n

        l_block_left = self.get_block_index_left(l)
        r_block_left = self.get_block_index_left(r)

        if l_block_left == r_block_left:
            res = self.lis[l]
            for k in range(l + 1, r + 1):
                res = self.op(res, self.lis[k])
            return res

        res = self.lis[l]
        for i in range(l + 1, min((l_block_left + 1) * self.block_size, self.n)):
            res = self.op(res, self.lis[i])

        for block_ind in range(l_block_left + 1, r_block_left):
            res = self.op(res, self.blocks[block_ind])

        for i in range(r_block_left * self.block_size, r + 1):
            res = self.op(res, self.lis[i])

        return res

    def update(self, i: int, x: Any) -> None:
        assert 0 <= i < self.n
        self.lis[i] = x
        block_ind = self.get_block_index_left(i)
        start = block_ind * self.block_size
        end = min(start + self.block_size, self.n)
        if start < self.n:
            self.blocks[block_ind] = self.lis[start]
            for j in range(start + 1, end):
                self.blocks[block_ind] = self.op(self.blocks[block_ind], self.lis[j])

    def get(self, i: int) -> Any:
        assert 0 <= i < self.n
        return self.lis[i]


class SquareDivisionSpeedy(SquareDivision):
    def __init__(
        self,
        lis: list[Any],
        op: Callable[[Any, Any], Any],
        delete: Callable[[Any, Any], Any],
    ) -> None:
        """その値を削除する関数がある場合の平方分割ライブラリ

        更新は高速だがクエリがボトルネックなのであまり変わらない
        """
        self.delete = delete
        super().__init__(lis, op)

    def update(self, i: int, x: Any) -> None:
        assert 0 <= i < self.n

        block_ind = self.get_block_index_left(i)
        self.blocks[block_ind] = self.delete(self.blocks[block_ind], self.lis[i])
        self.lis[i] = x
        self.blocks[block_ind] = self.op(self.blocks[block_ind], self.lis[i])
