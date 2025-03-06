# competitive-verifier: PROBLEM https://judge.u-aizu.ac.jp/onlinejudge/description.jsp?id=GRL_1_A
from libs.dijkstra import dijkstra
from libs.graph import GraphW
from libs.standard_input import il
from libs.utils import INF

N, M, S = il()
G = GraphW(N, dire=True)

for _ in [0] * M:
    a, b, w = il()
    G.new_side(a, b, w)

ans = dijkstra(G.all(), S)

for t in ans:
    if t == INF:
        print("INF")
    else:
        print(t)
