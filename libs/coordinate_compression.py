from typing import List, Tuple


def coordinate_compression(lis: List[int] | Tuple[int]) -> List[int]:
    """
    座標圧縮します
    計算量は、O(N log N)です
    """
    res = []
    d = {num: ind for ind, num in enumerate(sorted(set(lis)))}

    for a in lis:
        res.append(d[a])

    return res
