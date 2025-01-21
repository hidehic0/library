def mod_add(a: int, b: int, mod: int):
    return (a + b) % mod


def mod_sub(a: int, b: int, mod: int):
    return (a - b) % mod


def mod_mul(a: int, b: int, mod: int):
    return (a * b) % mod


def mod_div(a: int, b: int, mod: int):
    return (a * pow(b, mod - 2, mod)) % mod


class ModInt:
    def __init__(self, x: int, mod: int = 998244353) -> None:
        self.x = x % mod
        self.mod = mod

    def val(self):
        return self.x

    def rhs(self, rhs) -> int:
        return rhs.x if isinstance(rhs, ModInt) else rhs

    def __add__(self, rhs) -> int:
        return mod_add(self.x, self.rhs(rhs), self.mod)

    def __iadd__(self, rhs) -> "ModInt":
        self.x = self.__add__(rhs)

        return self

    def __sub__(self, rhs) -> int:
        return mod_sub(self.x, self.rhs(rhs), self.mod)

    def __isub__(self, rhs) -> "ModInt":
        self.x = self.__sub__(rhs)

        return self

    def __mul__(self, rhs):
        return mod_mul(self.x, self.rhs(rhs), self.mod)

    def __imul__(self, rhs):
        self.x = self.__mul__(rhs)

        return self

    def __truediv__(self, rhs):
        return mod_div(self.x, self.rhs(rhs), self.mod)

    def __itruediv__(self, rhs):
        self.x = self.__truediv__(rhs)

        return self

    def __floordiv__(self, rhs):
        return (self.x // self.rhs(rhs)) % self.mod

    def __ifloordiv__(self, rhs):
        self.x = self.__floordiv__(rhs)

        return self

    def __pow__(self, rhs):
        return pow(self.x, self.rhs(rhs), self.mod)

    def __eq__(self, rhs) -> bool:
        return self.rhs(rhs) == self.x

    def __ne__(self, rhs) -> bool:
        return self.rhs(rhs) != self.x
