# DPのテンプレート
from typing import List


def partial_sum_dp(lis: List[int], X: int) -> List[bool]:
    """
    部分和dpのテンプレート
    lisは品物です
    dp配列の長さは、Xにします
    計算量は、O(X*len(L))みたいな感じ

    返り値は、dp配列で中身は到達できたかを、示すboolです
    """
    dp = [False] * (X + 1)
    dp[0] = True

    for a in lis:
        for k in reversed(range(len(dp))):
            if not dp[k]:
                continue

            if k + a >= len(dp):
                continue

            dp[k + a] = True

    return dp


def knapsack_dp(lis: list[list[int]], W: int) -> int:
    """
    ナップサック問題を一次元DPで解く
    lis: 品物のリスト [[重さ, 価値], ...]
    W: ナップサックの容量
    戻り値: 最大価値
    """
    if W < 0 or not lis:
        return 0

    dp = [0] * (W + 1)

    for w, v in lis:
        if w < 0 or v < 0:
            raise ValueError("Weight and value must be non-negative")
        for k in reversed(range(W - w + 1)):
            dp[k + w] = max(dp[k + w], dp[k] + v)

    return dp[W]


def article_breakdown(lis: List[List[int]]) -> List[List[int]]:
    """
    個数制限付きナップサックの品物を分解します
    個数の値が、各品物の一番右にあれば正常に動作します
    """
    res = []
    for w, v, c in lis:
        k = 1
        while c > 0:
            res.append([w * k, v * k])
            c -= k
            k = min(2 * k, c)

    return res
