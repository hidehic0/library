from typing import List, Tuple


def coordinates_to_id(H: int, W: int) -> Tuple[List[List[int]], List[Tuple[int]]]:
    """
    座標にID変換します

    返り値は、
    最初のが、座標からid
    二つめのが、idから座標
    です
    """
    ItC = [[-1] * W for _ in [0] * H]
    CtI = [(-1, -1) for _ in [0] * (H * W)]

    i = 0

    for x in range(H):
        for y in range(W):
            CtI[x][y] = i
            ItC[i] = (x, y)
            i += 1

    return CtI, ItC
