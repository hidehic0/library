from typing import Callable


def binary_search(fn: Callable[[int], bool], right: int = 0, left: int = -1) -> int:
    while right - left > 1:
        mid = (left + right) // 2

        if fn(mid):
            left = mid
        else:
            right = mid

    return left
