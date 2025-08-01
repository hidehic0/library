# competitive-verifier: PROBLEM https://judge.yosupo.jp/problem/enumerate_palindromes
from libs.manacher import manacher_algorithm

S = input()
T = ["$"]
for s in S:
    T.append(s)
    T.append("$")
T = "".join(T)

L = manacher_algorithm(T)

print(*[c - 1 if i % 2 == 0 else c - 1 for i, c in enumerate(L)][1:-1])
