from typing import List


class RollingHash:
    string: str
    mod: int
    base: int
    n: int

    def __init__(self, string: str, mod: int = (1 << 61) - 1) -> None:
        """
        RollingHash構造体
        衝突する可能性があるのでmodが違う二つで比較するのが有効

        string: 文字列
        mod: mod デフォルト値は2^61 - 1
        """
        self.string = string
        self.mod = mod
        self.base = len(set(string))

        self.n = n = len(string)
        self.pow = [1] * (n + 1)
        self.hash = [0] * (n + 1)

        for i in range(n):
            self.hash[i + 1] = (self.hash[i] * self.base + ord(string[i])) % mod

        for i in range(n):
            self.pow[i + 1] = self.pow[i] * self.base % mod

    def get(self, l: int, r: int) -> int:
        """
        lからrまでのハッシュ値を取得する
        0-indexed
        """
        return (self.hash[r] - self.hash[l] * self.pow[r - l]) % self.mod

    def lcp(self, b: int, bn: int) -> int:
        """
        2つのRollingHashの最長共通接頭辞を返す
        bがhashでbnがそのhashの長さです
        """

        left, right = 0, min(self.n, bn)

        while right - left > 1:
            mid = (left + right) // 2

            if self.get(0, mid) == b:
                left = mid
            else:
                right = mid

        return left
