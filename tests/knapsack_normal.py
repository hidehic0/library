# competitive-verifier: PROBLEM https://judge.u-aizu.ac.jp/onlinejudge/description.jsp?id=DPL_1_C
from libs.dp import knapsack_dp
from libs.standard_input import *


def main():
    N, W = il()
    L = [il()[::-1] for _ in [0] * N]
    NL = []

    for w, v in L:
        i = 1

        while w * i <= W:
            NL.append((w * i, v * i))
            i += 1

    print(knapsack_dp(NL, W))


if __name__ == "__main__":
    main()
