# competitive-verifier: PROBLEM https://judge.u-aizu.ac.jp/onlinejudge/description.jsp?id=GRL_1_A
from libs.dijkstra import dijkstra
from libs.graph import GraphW
from libs.standard_input import *

N, M, S = il()
G = GraphW(N)

for _ in [0] * M:
    a, b, w = il()
    G.new_side(a, b, w)

print(*dijkstra(G.all(), S), sep="\n")
