# DPのテンプレート
from typing import List


def partial_sum_dp(lis: List[int]) -> List[bool]:
    """
    部分和dpのテンプレート
    lisは品物です
    dp配列の長さは、sum(L)にします
    計算量は、O(sum(L)*len(L))みたいな感じ

    返り値は、dp配列で中身は到達できたかを、示すboolです
    """
    dp = [False] * (sum(lis) + 100)
    dp[0] = True

    for a in lis:
        for k in reversed(range(len(dp))):
            if not dp[k]:
                continue

            dp[k + a] = True

    return dp
