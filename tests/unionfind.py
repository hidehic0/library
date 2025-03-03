# competitive-verifier: PROBLEM https://judge.yosupo.jp/problem/unionfind
from libs.unionfind import UnionFind
from libs.standard_input import *
from libs.yn_func import *

N, Q = il()
UF = UnionFind(N)

while Q:
    t, u, v = il()

    if t == 0:
        UF.unite(u, v)
    else:
        print(int(UF.same(u, v)))

    Q -= 1
