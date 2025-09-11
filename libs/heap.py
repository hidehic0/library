from collections.abc import Callable
from typing import Any


def _keys_for_heapq(x: Any):
    """先頭の値を取得する"""
    cur = x

    while True:
        try:
            cur = cur[0]
        except TypeError:
            break

    return cur


class HeapBase:
    def __init__(
        self,
        arr: list[Any] = [],
        key: Callable[[Any], Any] = _keys_for_heapq,
    ) -> None:
        """arrはソート済みが前提です"""
        self.key: Callable[Any, Any] = key
        self.lis: list[tuple[Any, Any]] = [(self.key(x), x) for x in arr]

    def _op(self, a: int, b: int) -> bool:
        # aが親 bが子って感じだよ
        assert 0 <= a < b < len(self.lis)
        return True

    def push(self, x: Any) -> None:
        self.lis.append((self.key(x), x))
        i = len(self.lis) - 1
        while i != 0:
            p = (i - 1) // 2
            if self._op(p, i):
                self.lis[i], self.lis[p] = self.lis[p], self.lis[i]
                i = p
            else:
                break

    def pop(self) -> Any:
        assert len(self.lis) > 0
        res = self.lis[0][1]  # Return the original value (not the key)
        self.lis[0] = self.lis[-1]  # Move the last element to the root
        self.lis.pop()  # Remove the last element

        if not self.lis:  # If the heap is empty, return early
            return res

        # Restore heap property by sifting down
        i = 0
        while i * 2 + 1 < len(self.lis):  # While there is at least one child
            c1 = i * 2 + 1  # Left child
            c2 = i * 2 + 2  # Right child

            # Pick the smaller of the two children (if right child exists)
            smallest = c1
            if c2 < len(self.lis) and self._op(c1, c2):
                smallest = c2

            # If the parent is larger than the smallest child, swap
            if self._op(i, smallest):
                self.lis[i], self.lis[smallest] = self.lis[smallest], self.lis[i]
                i = smallest
            else:
                break

        return res

    def __len__(self) -> int:
        return len(self.lis)

    def __getitem__(self, i: int):
        return self.lis[i][1]


class HeapMin(HeapBase):
    def _op(self, a: int, b: int) -> bool:
        return self.lis[a][0] > self.lis[b][0]


class HeapMax(HeapBase):
    def _op(self, a: int, b: int) -> bool:
        return self.lis[a][0] < self.lis[b][0]
