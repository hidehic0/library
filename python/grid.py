def coordinate_check(x: int, y: int, H: int, W: int) -> bool:
    """
    座標がグリッドの範囲内にあるかチェックする関数
    0-indexedが前提
    """

    return 0 <= x < H and 0 <= y < W


from typing import List, Tuple


def grid_moves(
    x: int,
    y: int,
    H: int,
    W: int,
    moves: List[Tuple[int]] = [(0, 1), (0, -1), (1, 0), (-1, 0)],
    *check_funcs,
) -> List[Tuple[int]]:
    """
    現在の座標から、移動可能な座標をmovesをもとに列挙します。
    xとyは現在の座標
    HとWはグリッドのサイズ
    movesは移動する座標がいくつかを保存する
    check_funcsは、その座標の点が#だとかを自前で実装して判定はこちらでするみたいな感じ
    なおcheck_funcsは引数がxとyだけというのが条件
    """
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
