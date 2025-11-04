def mat_mul(
    a: list[list[int]],
    b: list[list[int]],
    mod: int | None = None,
) -> list[list[int]]:
    """行列の積"""
    res = [[0] * len(b[0]) for _ in [0] * len(a)]

    for i in range(len(a)):
        for j in range(len(b[0])):
            for k in range(len(b)):
                res[i][j] += a[i][k] * b[k][j]

                if mod is not None:
                    res[i][j] %= mod

    return res


def mat_pow(a: list[list[int]], n: int, mod: int | None = None) -> list[list[int]]:
    """行列累乗"""
    assert n >= 0

    res = [[0] * len(a) for _ in [0] * len(a)]

    for i in range(len(a)):
        res[i][i] = 1

    while n > 0:
        if n & 1:
            res = mat_mul(res, a) if mod is None else mat_mul(res, a, mod)

        a = mat_mul(a, a) if mod is None else mat_mul(a, a, mod)
        n >>= 1

    return res
