import heapq
from typing import List, Tuple


def dijkstra(
    graph: List[List[Tuple[int]]], startpoint: int = 0, output_prev: bool = False
) -> List[int] | Tuple[List[int], List[int]]:
    """
    ダイクストラ法です
    GraphW構造体を使う場合は、allメソッドで、そんまま入れてください
    定数倍速いのかは分かりません(いつも使っているフォーマット)
    経路復元したい場合は、output_prevをTrueにすればprevも返ってくるので、それを使用して復元してください
    0-indexedが前提です
    """
    used = [1 << 63] * len(graph)
    prev = [-1] * len(graph)
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
            prev[nxt] = cur

            heapq.heappush(PQ, (new_cos, nxt))

    if not output_prev:
        return used
    else:
        return used, prev


def dijkstra_getpath(prev_lis: List[int], goal_point: int) -> List[int]:
    """
    dijkstraの経路復元をします
    先述のdijkstra関数で、output_prevをTrueにして返ってきた、prevを引数として用います
    """
    res = []
    cur = goal_point

    while cur != -1:
        res.append(cur)
        cur = prev_lis[cur]

    return res[::-1]
