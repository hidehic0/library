# github link: https://github.com/hidehic0/library/blob/main/library_for_joi.py
import bisect
import heapq
import math
import sys
from collections import Counter, defaultdict, deque
from collections.abc import Callable
from itertools import accumulate, combinations, permutations
from typing import Any

sys.setrecursionlimit(5 * 10**5)


def s() -> str:
    """一行に一つのstringをinput"""
    return input()


def sl() -> list[str]:
    """一行に複数のstringをinput"""
    return s().split()


def ii() -> int:
    """一つのint"""
    return int(s())


def il(add_num: int = 0) -> list[int]:
    """一行に複数のint"""
    return list(map(lambda i: int(i) + add_num, sl()))


def li(n: int, func, *args: list[Any]) -> list[list[Any]]:
    """複数行の入力をサポート"""
    return [func(*args) for _ in [0] * n]


def YesNoTemplate(state: bool, upper: bool = False) -> str:
    """YesNo関数のテンプレート

    stateがTrueなら、upperに応じてYes,YESをreturn
    stateがFalseなら、upperに応じてNo,NOをreturnする
    """
    YES = ["Yes", "YES"]
    NO = ["No", "NO"]

    if state:
        return YES[int(upper)]
    else:
        return NO[int(upper)]


def YN(state: bool, upper: bool = False) -> None:
    """
    先程のYesNoTemplate関数の結果を出力する
    """
    res = YesNoTemplate(state, upper)

    print(res)


def YE(state: bool, upper: bool = False) -> bool | None:
    """
    boolがTrueならYesを出力してexit
    """

    if not state:
        return False

    YN(True, upper)
    exit()


def NE(state: bool, upper: bool = False) -> bool | None:
    """
    boolがTrueならNoを出力してexit
    """

    if not state:
        return False

    YN(False, upper)
    exit()


def coordinate_check(x: int, y: int, H: int, W: int) -> bool:
    """座標がグリッドの範囲内にあるかチェックする関数

    0-indexedが前提
    """
    return 0 <= x < H and 0 <= y < W


def grid_moves(
    x: int,
    y: int,
    H: int,
    W: int,
    moves: list[tuple[int]] | None = None,
    *check_funcs: list[Callable[[int, int], bool]],
) -> list[tuple[int]]:
    """現在の座標から、移動可能な座標をmovesをもとに列挙します。

    xとyは現在の座標
    HとWはグリッドのサイズ
    movesは移動する座標がいくつかを保存する
    check_funcsは、その座標の点が#だとかを自前で実装して判定はこちらでするみたいな感じ
    なおcheck_funcsは引数がxとyだけというのが条件
    追加の判定関数は、弾く場合は、False それ以外ならTrueで
    """
    if moves is None:
        moves = ([(0, 1), (0, -1), (1, 0), (-1, 0)],)

    res = []

    for mx, my in moves:
        nx, ny = x + mx, y + my

        if not coordinate_check(nx, ny, H, W):
            continue

        for f in check_funcs:
            if not f(nx, ny):
                break
        else:
            res.append((nx, ny))

    return res


def euclid_dis(x1: int, y1: int, x2: int, y2: int) -> int:
    """ユークリッド距離を計算する関数

    注意:
    この関数はsqrtを取りません(主に少数誤差用)
    sqrtを取りたい場合は、自分で計算してください
    """
    return ((x1 - x2) ** 2) + ((y1 - y2) ** 2)


def manhattan_dis(x1: int, y1: int, x2: int, y2: int) -> int:
    """マンハッタン距離を計算する関数"""
    return abs(x1 - x2) + abs(y1 - y2)


def manhattan_45turn(x: int, y: int) -> tuple[int]:
    """マンハッタン距離用の座標を45度回転する関数

    回転すると、マンハッタン距離が、チェビシェフ距離になるので、距離の最大値などが簡単に求められます
    """
    res_x = x - y
    res_y = x + y

    return res_x, res_y


def chebyshev_dis(x1: int, y1: int, x2: int, y2: int) -> int:
    """チェビシェフ距離を計算する関数"""
    return max(abs(x1 - x2), abs(y1 - y2))


class ChangeMin:
    def __init__(self, x) -> None:
        """Change min構造体

        代入時現在の値より代入する値が低ければ代入される
        setメソッドで代入する
        """
        self.x = x

    def set(self, new) -> None:
        self.x = min(self.x, new)

    def val(self) -> any:
        return self.x


class ChangeMax:
    def __init__(self, x) -> None:
        """Change min構造体

        代入時現在の値より代入する値が大きければ代入される
        setメソッドで代入する
        """
        self.x = x

    def set(self, new) -> None:
        self.x = max(self.x, new)

    def val(self) -> any:
        return self.x


def binary_search(
    fn: Callable[[int], bool],
    right: int = 0,
    left: int = -1,
    return_left: bool = True,
) -> int:
    """二分探索の抽象的なライブラリ

    評価関数の結果に応じて、二分探索する
    最終的にはleftを出力します

    関数のテンプレート
    def check(mid:int):
        if A[mid] > x:
            return True
        else:
            return False

    midは必須です。それ以外はご自由にどうぞ
    """
    while right - left > 1:
        mid = (left + right) // 2

        if fn(mid):
            left = mid
        else:
            right = mid

    return left if return_left else right


def compress_1d(l: list[Any]) -> dict[Any, int]:
    """一次元座圧"""
    return {k: v for v, k in enumerate(sorted(set(l)))}


INF = 1 << 63
lowerlist = list("abcdefghijklmnopqrstuvwxyz")
upperlist = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
MOVES1 = [(0, 1), (0, -1), (1, 0), (-1, 0)]
MOVES2 = MOVES1 + [(1, 1), (1, -1), (-1, 1), (-1, -1)]
# ライブラリ終わり
