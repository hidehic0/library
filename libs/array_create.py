# 多次元配列作成
from typing import List, Any


def create_array2(a: int, b: int, default: Any = 0) -> List[List[Any]]:
    """
    ２次元配列を初期化する関数
    """
    return [[default] * b for _ in [0] * a]


def create_array3(a: int, b: int, c: int, default: Any = 0) -> List[List[List[Any]]]:
    """
    ３次元配列を初期化する関数
    """
    return [[[default] * c for _ in [0] * b] for _ in [0] * a]
