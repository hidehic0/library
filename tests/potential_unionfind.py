# competitive-verifier: PROBLEM https://judge.u-aizu.ac.jp/onlinejudge/description.jsp?id=DSL_1_B
from libs.potential_unionfind import PotentialUnionFind
from libs.standard_input import il

N, Q = il()
PUF = PotentialUnionFind(N)

while Q:
    l = il()

    if l[0] == 0:
        x, y, z = l[1:]

        PUF.unite(x, y, z)
    else:
        x, y = l[1:]
        if not PUF.same(x, y):
            print("?")
        else:
            print(PUF.diff(x, y))

    Q -= 1
