from typing import Tuple


def euclid_dis(x1: int, y1: int, x2: int, y2: int) -> int:
    """
    ユークリッド距離を計算します

    注意:
    この関数はsqrtを取りません(主に少数誤差用)
    sqrtを取りたい場合は、自分で計算してください
    """

    return ((x1 - x2) ** 2) + ((y1 - y2) ** 2)


def manhattan_dis(x1: int, y1: int, x2: int, y2: int) -> int:
    """
    マンハッタン距離を計算します
    """

    return abs(x1 - x2) + abs(y1 - y2)


def manhattan_45turn(x: int, y: int) -> Tuple[int]:
    """
    座標を45度回転します
    回転すると、マンハッタン距離が、チェビシェフ距離になるので、距離の最大値などが簡単に求められます
    """

    res_x = x - y
    res_y = x + y

    return res_x, res_y
