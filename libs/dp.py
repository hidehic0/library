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


def knapsack_dp(lis: List[List[int]], W: int) -> List[int]:
    """
    ナップサックdpのテンプレート
    lisは品物のリスト
    原則品物は、w,vの形で与えられ、wが重さ、vが価値、となる
    価値と重さを逆転させたい場合は自分でやってください
    dp配列は、定数倍高速化のため、一次元配列として扱う
    dp配列の長さは、Wとします
    """

    dp = [-(1 << 63)] * (W + 1)
    dp[0] = 0

    for w, v in lis:
        for k in reversed(range(len(dp))):
            if w + k >= len(dp):
                continue

            dp[w + k] = max(dp[w + k], dp[k] + v)

    return dp
