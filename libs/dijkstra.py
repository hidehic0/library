import heapq
from typing import List, Tuple


def dijkstra(graph: List[List[Tuple[int]]], startpoint: int = 0) -> List[int]:
    used = [1 << 63] * len(graph)
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
