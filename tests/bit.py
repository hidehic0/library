# competitive-verifier: PROBLEM https://judge.yosupo.jp/problem/point_add_range_sum
from libs.bit import BIT
from libs.standard_input import *

N, Q = il()
A = il()
bit = BIT(N)

for i in range(N):
    bit.add(i + 1, A[i])

while Q:
    lis = il()
    if lis[0] == 0:
        bit.add(lis[1] + 1, lis[2])
    else:
        print(bit.interval_sum(lis[1], lis[2]))

    Q -= 1
