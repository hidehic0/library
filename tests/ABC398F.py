from libs.rollinghash import RollingHash
import random

S = input()
N = len(S)

if len(S) == 1:
    print(S)
    exit()

H1 = RollingHash(S)
RH1 = RollingHash(S[::-1])

H2 = RollingHash(S, 998244353)
RH2 = RollingHash(S[::-1], 998244353)


for i in range(N + 1):
    if H1.get(i, N) == RH1.get(0, N - i) and H2.get(i, N) == RH2.get(0, N - i):
        print(S + S[:i][::-1])
        exit()
