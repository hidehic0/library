def coordinates_to_id(h: int, w: int) -> tuple[list[list[int]], list[tuple[int, int]]]:
    """座標を一次元のindexに変換する関数

    返り値は、
    最初のが、座標からid
    二つめのが、idから座標
    です
    """
    ItC = [[-1] * w for _ in [0] * h]
    CtI = [(-1, -1) for _ in [0] * (h * w)]

    i = 0

    for x in range(h):
        for y in range(w):
            ItC[x][y] = i
            CtI[i] = (x, y)
            i += 1

    return CtI, ItC
