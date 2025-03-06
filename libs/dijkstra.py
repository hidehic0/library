import heapq
from typing import List, Tuple


def dijkstra(graph: List[List[Tuple[int]]], startpoint: int = 0) -> List[int]:
    """
    ダイクストラ法です
    GraphW構造体を使う場合は、allメソッドで、そんまま入れてください
    定数倍速いのかは分かりません(いつも使っているフォーマット)
    0-indexedが前提です
    """
    used = [1 << 63] * len(graph)
    if not 0 <= startpoint < len(graph):
        raise IndexError("あのー0-indexedですか?")
    used[startpoint] = 0
    PQ = [(0, startpoint)]

    while PQ:
        cos, cur = heapq.heappop(PQ)

        if used[cur] < cos:
            continue

        for nxt, w in graph[cur]:
            new_cos = cos + w

            if new_cos >= used[nxt]:
                continue

            used[nxt] = new_cos

            heapq.heappush(PQ, (new_cos, nxt))

    return used
