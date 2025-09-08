from libs.prefix_sum_2d import PrefixSum2D

H, W = map(int, input().split())
K = int(input())
S = [input() for _ in [0] * H]

JS, IS, OS = PrefixSum2D(H, W), PrefixSum2D(H, W), PrefixSum2D(H, W)

for i in range(H):
    for k in range(W):
        match S[i][k]:
            case "J":
                JS.add(i, k, 1)

            case "O":
                OS.add(i, k, 1)

            case "I":
                IS.add(i, k, 1)

JS.build()
IS.build()
OS.build()

for _ in [0] * K:
    ax, ay, bx, by = map(int, input().split())
    ax -= 1
    ay -= 1
    bx -= 1
    by -= 1

    print(JS.prod(ax, ay, bx, by), OS.prod(ax, ay, bx, by), IS.prod(ax, ay, bx, by))
