import math
from typing import Any, Callable, List


def mo_algorithm(
    N: int,
    queries: List[Any],
    add: Callable[[int], Any],
    delete: Callable[[int], Any],
    getvalue: Callable[[], Any],
) -> List[Any]:
    """
    Mo's algorithmの関数
    queriesは、(左端, 右端)で1-indexed
    addはあるindexが追加される時の値を現在の値にする
    deleteはあるindexが削除される時の値を現在の値にする
    getvalueは現在の値を返す
    """
    Q = len(queries)
    res = [None] * Q
    M = int(max(1, 1.0 * N / max(1, math.sqrt(Q * 2.0 / 3.0))))

    queries = [(l, r, i) for i, (l, r) in enumerate(queries)]
    queries.sort(key=lambda x: (x[0] // M, x[1] if (x[0] // M) % 2 == 0 else -x[1]))

    cl, cr = 0, -1

    for l, r, ind in queries:
        l -= 1
        r -= 1
        while cl > l:
            cl -= 1
            add(cl)

        while cr < r:
            cr += 1
            add(cr)

        while cl < l:
            delete(cl)
            cl += 1

        while cr > r:
            delete(cr)
            cr -= 1

        res[ind] = getvalue()

    return res
