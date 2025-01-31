def euclid_dis(x1: int, y1: int, x2: int, y2: int) -> int:
    """
    ユークリッド距離を計算します

    注意:
    この関数はsqrtを取りません(主に少数誤差用)
    sqrtを取りたい場合は、自分で計算してください
    """

    return ((x1 - x2) ** 2) + ((y1 - y2) ** 2)
