# インタラクティブ問題の時はIS_INTERACTIVEをTrueにしましょう

# 標準入力関数
import io
import os
import sys
from typing import Any, List

if sys.argv[0] == "Main.py":
    try:
        # ruff: noqa
        if not IS_INTERACTIVE:
            input = io.BytesIO(os.read(0, os.fstat(0).st_size)).readline
    except:
        pass


def s() -> str:
    """
    一行に一つのstringをinput
    """
    return input()


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
