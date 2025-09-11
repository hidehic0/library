# 多次元配列作成
from typing import Any


def create_array1(n: int, default=0) -> list[Any]:
    """1次元配列を初期化する関数"""
    return [default] * n


def create_array2(a: int, b: int, default=0) -> list[list[Any]]:
    """2次元配列を初期化する関数"""
    return [[default] * b for _ in [0] * a]


def create_array3(a: int, b: int, c: int, default=0) -> list[list[list[Any]]]:
    """3次元配列を初期化する関数"""
    return [[[default] * c for _ in [0] * b] for _ in [0] * a]
