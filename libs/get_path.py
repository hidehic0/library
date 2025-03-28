from typing import List


def getpath(prev_lis: List[int], goal_point: int) -> List[int]:
    """
    経路復元をします
    dijkstra関数を使う場合、output_prevをTrueにして返ってきた、prevを引数として用います
    他の場合は、移動の時、usedを付けるついでに、prevに現在の頂点を付けてあげるといいです
    """
    res = []
    cur = goal_point

    while cur != -1:
        res.append(cur)
        cur = prev_lis[cur]

    return res[::-1]
