from typing import Any, Callable, List


def rerooting(
    G: List[List[int]],
    merge: Callable[[Any, Any], Any],
    add_root: Callable[[int, Any], Any],
    e,
) -> List[Any]:
    _n = len(G)
    dp: List[List[Any]] = [[]] * _n
    ans: List[Any] = [e] * _n

    def _dfs(u: int, p: int = -1):
        nonlocal dp, merge, add_root, e

        res: Any = e
        dp[u] = [e] * (len(G[u]))

        for i, v in enumerate(G[u]):
            if v == p:
                continue

            dp[u][i] = _dfs(v, u)
            res = merge(res, dp[u][i])

        return add_root(u, res)

    def _bfs(u: int, cur: Any, p: int = -1):
        nonlocal dp, merge, add_root, e, ans
        deg = len(G[u])

        for i in range(deg):
            if G[u][i] == p:
                dp[u][i] = cur

        dp_l, dp_r = [e] * (deg + 1), [e] * (deg + 1)

        for i in range(deg):
            dp_l[i + 1] = merge(dp_l[i], dp[u][i])

        for i in reversed(range(deg)):
            dp_r[i] = merge(dp_r[i + 1], dp[u][i])

        ans[u] = add_root(u, dp_l[deg])

        for i in range(deg):
            if G[u][i] != p:
                _bfs(G[u][i], add_root(u, merge(dp_l[i], dp_r[i + 1])), u)

    _dfs(0)
    _bfs(0, e)

    return ans
