def mod_add(a: int, b: int, mod: int):
    return (a + b) % mod


def mod_sub(a: int, b: int, mod: int):
    return (a - b) % mod


def mod_mul(a: int, b: int, mod: int):
    return (a * b) % mod


def mod_div(a: int, b: int, mod: int):
    return a * pow(b, mod - 2, mod)
