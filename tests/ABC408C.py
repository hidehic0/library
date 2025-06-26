from libs.dual_segtree import DualSegmentTree

N, M = map(int, input().split())
seg = DualSegmentTree(lambda a, b: a + b, 0, N)

for _ in [0] * M:
    l, r = map(int, input().split())
    seg.apply(l - 1, r, 1)

ans = 1000000000000000000

for i in range(N):
    ans = min(ans, seg.get(i))

print(ans)
