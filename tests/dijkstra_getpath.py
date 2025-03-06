# competitive-verifier: PROBLEM https://judge.yosupo.jp/problem/shortest_path
from libs.dijkstra import dijkstra
from libs.get_path import getpath
from libs.graph import GraphW
from libs.standard_input import il
from libs.utils import INF

N, M, S, T = il()
G = GraphW(N, dire=True)

for _ in [0] * M:
    a, b, w = il()
    G.new_side(a, b, w)

used, prev = dijkstra(G.all(), S, True)

if used[T] == INF:
    print(-1)
else:
    path = getpath(prev, T)

    print(used[T], len(path) - 1)

    for i in range(len(path) - 1):
        print(path[i], path[i + 1])
