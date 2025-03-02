# competitive-verifier: PROBLEM https://judge.u-aizu.ac.jp/onlinejudge/description.jsp?id=DPL_1_B
from libs.dp import knapsack_dp
from libs.standard_input import *


def main():
    N, W = il()
    L = [il()[::-1] for _ in [0] * N]

    print(knapsack_dp(L, W))


if __name__ == "__main__":
    main()
