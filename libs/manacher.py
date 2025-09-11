def manacher_algorithm(S: str) -> list[int]:
    """Manacher algorithm

    res_i = S_iを中心とした最長の回文の半径
    """
    # いまいち原理は分からないけどうまいことメモ化してそう
    _n = len(S)
    res = [0] * _n

    i = k = 0

    while i < _n:
        while i - k >= 0 and i + k < _n and S[i - k] == S[i + k]:
            k += 1

        res[i] = k
        a = 1

        while i - a >= 0 and a + res[i - a] < k:
            res[i + a] = res[i - a]
            a += 1
        i += a
        k -= a

    return res
