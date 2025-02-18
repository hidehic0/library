# 標準入力関数
import sys
from typing import Any, List


def s() -> str:
    """
    一行に一つのstringをinput
    """
    return sys.stdin.readline().rstrip()


def sl() -> List[str]:
    """
    一行に複数のstringをinput
    """
    return s().split()


def ii() -> int:
    """
    一つのint
    """
    return int(s())


def il(add_num: int = 0) -> List[int]:
    """
    一行に複数のint
    """
    return list(map(lambda i: int(i) + add_num, sl()))


def li(n: int, func, *args) -> List[List[Any]]:
    """
    複数行の入力をサポート
    """
    return [func(*args) for _ in [0] * n]
