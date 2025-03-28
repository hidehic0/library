from typing import List, Tuple


def compress_1d(points: List[int] | Tuple[int]) -> List[int]:
    """
    一次元座標圧縮します
    計算量は、O(N log N)です

    lとrは、まとめて入れる事で、座圧できます
    """
    res = []
    d = {num: ind for ind, num in enumerate(sorted(set(points)))}

    for a in points:
        res.append(d[a])

    return res


def compress_2d(points):
    """
    2次元座標圧縮を行う関数
    入力: points - [(x1, y1), (x2, y2), ...] の形式の座標リスト
    出力: 圧縮後の座標リストと、元の座標から圧縮後の座標へのマッピング
    """
    # x座標とy座標を分離
    x_coords = sorted(set(x for x, y in points))  # 重複を削除してソート
    y_coords = sorted(set(y for x, y in points))

    # 座標から圧縮後の値へのマッピング辞書を作成
    x_map = {val: idx for idx, val in enumerate(x_coords)}
    y_map = {val: idx for idx, val in enumerate(y_coords)}

    # 圧縮後の座標リストを作成
    compressed = [(x_map[x], y_map[y]) for x, y in points]

    return compressed
