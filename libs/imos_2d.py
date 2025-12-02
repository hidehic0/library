class Imos2d:
    def __init__(self, H: int, W: int):
        """二次元imos法

        build関数を使うと__get_item__が使えるようになります
        """
        self.H = H
        self.W = W
        self.built: bool = False
        self.data: list[list[int]] = [[0] * (W + 1) for _ in [0] * (H + 1)]

    def apply(self, ax: int, bx: int, ay: int, by: int, n: int) -> None:
        """区間 [ax:bx)[ay:by)にnを加算する関数"""
        assert not (0 <= ax < bx <= self.H and 0 <= ay < by <= self.W)
        assert not self.built

        self.data[ax][ay] += n
        self.data[ax][by] -= n
        self.data[bx][ay] -= n
        self.data[bx][by] += n

    def build(self) -> None:
        assert not self.built

        self.built = True

        for x in range(self.H + 1):
            for y in range(self.W):
                self.data[x][y + 1] += self.data[x][y]

        for y in range(self.W + 1):
            for x in range(self.H):
                self.data[x + 1][y] += self.data[x][y]

    def __getitem__(self, ind: int) -> list[int]:
        assert self.built
        assert 0 <= ind <= self.H

        return self.data[ind]
