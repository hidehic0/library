class PrefixSum2D:
    def __init__(self, h: int, w: int) -> None:
        self.data = [[0] * (w + 1) for _ in [0] * (h + 1)]
        self.builded = False
        self.h = h
        self.w = w

    def add(self, x: int, y: int, a: int) -> None:
        assert 0 <= x < self.h and 0 <= y < self.w

        self.data[x + 1][y + 1] += a

    def build(self) -> None:
        assert not self.builded

        for i in range(self.h + 1):
            for k in range(self.w):
                self.data[i][k + 1] += self.data[i][k]

        for k in range(self.w + 1):
            for i in range(self.h):
                self.data[i + 1][k] += self.data[i][k]

    def prod(self, ax: int, ay: int, bx: int, by: int) -> int:
        assert 0 <= ax <= bx < self.h and 0 <= ay <= by < self.w

        return (
            self.data[bx + 1][by + 1]
            + self.data[ax][ay]
            - self.data[ax][by + 1]
            - self.data[bx + 1][ay]
        )
