r"""
 ______________________
< it's hidehico's code >
 ----------------------
   \
    \
        .--.
       |o_o |
       |:_/ |
      //   \ \
     (|     | )
    /'\_   _/`\
    \___)=(___/
"""

# ライブラリと関数と便利変数
# ライブラリ
import bisect
import copy
import heapq
import math
import sys
from collections import Counter, defaultdict, deque
from itertools import accumulate, combinations, permutations
from math import gcd, lcm, pi
from operator import itemgetter
from typing import Any, List, Tuple

# from atcoder.segtree import SegTree
# from atcoder.lazysegtree import LazySegTree
# from atcoder.dsu import DSU

# cortedcontainersは使うときだけ wandbox非対応なので
# from sortedcontainers import SortedDict, SortedSet, SortedList

# import pypyjit
# pypyjit.set_param("max_unroll_recursion=-1")

sys.setrecursionlimit(5 * 10**5)
import io
import os
import sys
from typing import Any, List

# インタラクティブ問題の時はIS_INTERACTIVEをTrueにしましょう
# IS_INTERACTIVE = False

# 標準入力関数
# if sys.argv[0] == "Main.py":
#     if not IS_INTERACTIVE:
#         input = io.BytesIO(os.read(0, os.fstat(0).st_size)).readline().decode().rstrip


def s() -> str:
    """
    一行に一つのstringをinput
    """
    return input()


def sl() -> List[str]:
    """
    一行に複数のstringをinput
    """
    return s().split()


def ii() -> int:
    """
    一つのint
    """
    return int(s())


def il(add_num: int = 0) -> List[int]:
    """
    一行に複数のint
    """
    return list(map(lambda i: int(i) + add_num, sl()))


def li(n: int, func, *args) -> List[List[Any]]:
    """
    複数行の入力をサポート
    """
    return [func(*args) for _ in [0] * n]


from typing import List


# 数学型関数
def is_prime(n: int) -> int:
    """
    素数判定します
    計算量は定数時間です。正確には、繰り返し二乗法の計算量によりです
    アルゴリズムはミラーラビンの素数判定を使用しています
    nが2^64を越えると動作しません
    """
    if n == 1:
        return False

    def f(a, t, n):
        x = pow(a, t, n)
        nt = n - 1
        while t != nt and x != 1 and x != nt:
            x = pow(x, 2, n)
            t <<= 1

        return t & 1 or x == nt

    if n == 2:
        return True
    elif n % 2 == 0:
        return False

    d = n - 1
    d >>= 1

    while d & 1 == 0:
        d >>= 1

    checklist = (
        [2, 7, 61] if 2**32 > n else [2, 325, 9375, 28178, 450775, 9780504, 1795265022]
    )

    for i in checklist:
        if i >= n:
            break
        if not f(i, d, n):
            return False

    return True


def eratosthenes(n: int) -> List[int]:
    """
    n以下の素数を列挙します
    計算量は、O(n log log n)です
    先程の素数判定法で列挙するよりも、少し速いです
    列挙した素数は昇順に並んでいます
    アルゴリズムはエラトステネスです
    """
    primes = [True] * (n + 1)
    primes[0], primes[1] = False, False
    i = 2
    while i**2 <= n:
        if primes[i]:
            for k in range(i * 2, n + 1, i):
                primes[k] = False

        i += 1

    return [i for i, p in enumerate(primes) if p]


def calc_divisors(n: int):
    """
    Nの約数列挙します
    計算量は、√Nです
    約数は昇順に並んでいます
    """
    result = []

    for i in range(1, n + 1):
        if i * i > n:
            break

        if n % i != 0:
            continue

        result.append(i)
        if n // i != i:
            result.append(n // i)

    return sorted(result)


def factorization(n: int) -> List[List[int]]:
    """
    nを素因数分解します
    計算量は、√Nです(要改善)
    複数回素因数分解を行なう場合は、√N以下の素数を列挙したので試し割りした法が速いです
    """
    result = []
    tmp = n
    for i in range(2, int(-(-(n**0.5) // 1)) + 1):
        if tmp % i == 0:
            cnt = 0
            while tmp % i == 0:
                cnt += 1
                tmp //= i
            result.append([i, cnt])

    if tmp != 1:
        result.append([tmp, 1])

    if result == []:
        result.append([n, 1])

    return result


def factorization_plural(L: List[int]) -> List[List[List[int]]]:
    """
    複数の数の素因数分解を行ないます
    計算量は、O(N * (√max(L) log log √max(L)))
    みたいな感じです

    最初に素数を列挙するため、普通の素因数分解より効率がいいです
    """
    res = []
    primes = eratosthenes(int(max(L) ** 0.5) + 20)

    def solve(n):
        t = []
        for p in primes:
            if n % p == 0:
                cnt = 0
                while n % p == 0:
                    cnt += 1
                    n //= p

                t.append([p, cnt])

        if n != 1:
            t.append([n, 1])

        if t == []:
            t.append([n, 1])

        return t

    for n in L:
        res.append(solve(n))

    return res


def simple_sigma(n: int) -> int:
    """
    1からnまでの総和を求める関数
    つまり和の公式
    """
    return (n * (n + 1)) // 2


def comb(n: int, r: int, mod: int | None = None) -> int:
    """
    高速なはずの二項係数
    modを指定すれば、mod付きになる
    """
    a = 1

    for i in range(n - r + 1, n + 1):
        a *= i

        if mod:
            a %= mod

    b = 1

    for i in range(1, r + 1):
        b *= i
        if mod:
            b %= mod

    if mod:
        return a * pow(b, -1, mod) % mod
    else:
        return a * b


# 多次元配列作成
from typing import List, Any


def create_array1(n: int, default: Any = 0) -> List[Any]:
    """
    1次元配列を初期化する関数
    """
    return [default] * n


def create_array2(a: int, b: int, default: Any = 0) -> List[List[Any]]:
    """
    2次元配列を初期化する関数
    """
    return [[default] * b for _ in [0] * a]


def create_array3(a: int, b: int, c: int, default: Any = 0) -> List[List[List[Any]]]:
    """
    3次元配列を初期化する関数
    """
    return [[[default] * c for _ in [0] * b] for _ in [0] * a]


from typing import Callable


def binary_search(
    fn: Callable[[int], bool], right: int = 0, left: int = -1, return_left: bool = True
) -> int:
    """
    二分探索の抽象的なライブラリ
    評価関数の結果に応じて、二分探索する
    最終的にはleftを出力します

    関数のテンプレート
    def check(mid:int):
        if A[mid] > x:
            return True
        else:
            return False

    midは必須です。それ以外はご自由にどうぞ
    """
    while right - left > 1:
        mid = (left + right) // 2

        if fn(mid):
            left = mid
        else:
            right = mid

    return left if return_left else right


def mod_add(a: int, b: int, mod: int):
    """
    足し算してmodを取った値を出力
    O(1)
    """
    return (a + b) % mod


def mod_sub(a: int, b: int, mod: int):
    """
    引き算してmodを取った値を出力
    O(1)
    """
    return (a - b) % mod


def mod_mul(a: int, b: int, mod: int):
    """
    掛け算してmodを取った値を出力
    O(1)
    """
    return (a * b) % mod


def mod_div(a: int, b: int, mod: int):
    """
    割り算してmodを取った値を出力
    フェルマーの小定理を使って計算します
    O(log mod)
    """
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


# YesNo関数
def YesNoTemplate(state: bool, upper: bool = False) -> str:
    """
    stateがTrueなら、upperに応じてYes,YESをreturn
    stateがFalseなら、upperに応じてNo,NOをreturnする
    """
    YES = ["Yes", "YES"]
    NO = ["No", "NO"]

    if state:
        return YES[int(upper)]
    else:
        return NO[int(upper)]


def YN(state: bool, upper: bool = False) -> None:
    """
    先程のYesNoTemplate関数の結果を出力する
    """
    res = YesNoTemplate(state, upper)

    print(res)


def YE(state: bool, upper: bool = False) -> bool | None:
    """
    boolがTrueならYesを出力してexit
    """

    if not state:
        return False

    YN(True, upper)
    exit()


def NE(state: bool, upper: bool = False) -> bool | None:
    """
    boolがTrueならNoを出力してexit
    """

    if not state:
        return False

    YN(False, upper)
    exit()


def coordinate_check(x: int, y: int, H: int, W: int) -> bool:
    """
    座標がグリッドの範囲内にあるかチェックする関数
    0-indexedが前提
    """

    return 0 <= x < H and 0 <= y < W


from typing import List, Tuple


def grid_moves(
    x: int,
    y: int,
    H: int,
    W: int,
    moves: List[Tuple[int]] = [(0, 1), (0, -1), (1, 0), (-1, 0)],
    *check_funcs,
) -> List[Tuple[int]]:
    """
    現在の座標から、移動可能な座標をmovesをもとに列挙します。
    xとyは現在の座標
    HとWはグリッドのサイズ
    movesは移動する座標がいくつかを保存する
    check_funcsは、その座標の点が#だとかを自前で実装して判定はこちらでするみたいな感じ
    なおcheck_funcsは引数がxとyだけというのが条件
    追加の判定関数は、弾く場合は、False それ以外ならTrueで
    """
    res = []

    for mx, my in moves:
        nx, ny = x + mx, y + my

        if not coordinate_check(nx, ny, H, W):
            continue

        for f in check_funcs:
            if not f(nx, ny):
                break
        else:
            res.append((nx, ny))

    return res


from typing import List, Tuple


def coordinates_to_id(H: int, W: int) -> Tuple[List[List[int]], List[Tuple[int]]]:
    """
    座標にID変換します

    返り値は、
    最初のが、座標からid
    二つめのが、idから座標
    です
    """
    ItC = [[-1] * W for _ in [0] * H]
    CtI = [(-1, -1) for _ in [0] * (H * W)]

    i = 0

    for x in range(H):
        for y in range(W):
            ItC[x][y] = i
            CtI[i] = (x, y)
            i += 1

    return CtI, ItC


import heapq
from typing import List, Tuple


def dijkstra(
    graph: List[List[Tuple[int]]], startpoint: int = 0, output_prev: bool = False
) -> List[int] | Tuple[List[int], List[int]]:
    """
    ダイクストラ法です
    GraphW構造体を使う場合は、allメソッドで、そんまま入れてください
    定数倍速いのかは分かりません(いつも使っているフォーマット)
    経路復元したい場合は、output_prevをTrueにすればprevも返ってくるので、それを使用して復元してください
    0-indexedが前提です
    """
    used = [1 << 63] * len(graph)
    prev = [-1] * len(graph)
    if not 0 <= startpoint < len(graph):
        raise IndexError("あのー0-indexedですか?")
    used[startpoint] = 0
    PQ = [(0, startpoint)]

    while PQ:
        cos, cur = heapq.heappop(PQ)

        if used[cur] < cos:
            continue

        for nxt, w in graph[cur]:
            new_cos = cos + w

            if new_cos >= used[nxt]:
                continue

            used[nxt] = new_cos
            prev[nxt] = cur

            heapq.heappush(PQ, (new_cos, nxt))

    if not output_prev:
        return used
    else:
        return used, prev


from typing import List


def getpath(prev_lis: List[int], goal_point: int) -> List[int]:
    """
    経路復元をします
    dijkstra関数を使う場合、output_prevをTrueにして返ってきた、prevを引数として用います
    他の場合は、移動の時、usedを付けるついでに、prevに現在の頂点を付けてあげるといいです
    """
    res = []
    cur = goal_point

    while cur != -1:
        res.append(cur)
        cur = prev_lis[cur]

    return res[::-1]


# DPのテンプレート
from typing import List


def partial_sum_dp(lis: List[int], X: int) -> List[bool]:
    """
    部分和dpのテンプレート
    lisは品物です
    dp配列の長さは、Xにします
    計算量は、O(X*len(L))みたいな感じ

    返り値は、dp配列で中身は到達できたかを、示すboolです
    """
    dp = [False] * (X + 1)
    dp[0] = True

    for a in lis:
        for k in reversed(range(len(dp))):
            if not dp[k]:
                continue

            if k + a >= len(dp):
                continue

            dp[k + a] = True

    return dp


def knapsack_dp(lis: list[list[int]], W: int) -> int:
    """
    ナップサック問題を一次元DPで解く
    lis: 品物のリスト [[重さ, 価値], ...]
    W: ナップサックの容量
    戻り値: 最大価値
    """
    if W < 0 or not lis:
        return 0

    dp = [0] * (W + 1)

    for w, v in lis:
        if w < 0 or v < 0:
            raise ValueError("Weight and value must be non-negative")
        for k in reversed(range(W - w + 1)):
            dp[k + w] = max(dp[k + w], dp[k] + v)

    return dp[W]


def article_breakdown(lis: List[List[int]]) -> List[List[int]]:
    """
    個数制限付きナップサックの品物を分解します
    個数の値が、各品物の一番右にあれば正常に動作します
    """
    res = []
    for w, v, c in lis:
        k = 1
        while c > 0:
            res.append([w * k, v * k])
            c -= k
            k = min(2 * k, c)

    return res


from typing import List, Tuple


def compress_1d(points: List[int] | Tuple[int]) -> List[int]:
    """
    一次元座標圧縮します
    計算量は、O(N log N)です

    lとrは、まとめて入れる事で、座圧できます
    """
    res = []
    d = {num: ind for ind, num in enumerate(sorted(set(points)))}

    for a in points:
        res.append(d[a])

    return res


def compress_2d(points):
    """
    2次元座標圧縮を行う関数
    入力: points - [(x1, y1), (x2, y2), ...] の形式の座標リスト
    出力: 圧縮後の座標リストと、元の座標から圧縮後の座標へのマッピング
    """
    # x座標とy座標を分離
    x_coords = sorted(set(x for x, y in points))  # 重複を削除してソート
    y_coords = sorted(set(y for x, y in points))

    # 座標から圧縮後の値へのマッピング辞書を作成
    x_map = {val: idx for idx, val in enumerate(x_coords)}
    y_map = {val: idx for idx, val in enumerate(y_coords)}

    # 圧縮後の座標リストを作成
    compressed = [(x_map[x], y_map[y]) for x, y in points]

    return compressed


# ac_libraryのメモ
"""
segtree

初期化するとき
Segtree(op,e,v)

opはマージする関数
例

def op(a,b):
    return a+b

eは初期化する値

vは配列の長さまたは、初期化する内容
"""
from collections import defaultdict
import math


class WeightedTreeLCA:
    def __init__(self, n):
        """初期化: ノード数nの木を構築（0-indexed）"""
        self.n = n
        self.log = math.ceil(math.log2(n)) + 1
        self.adj = defaultdict(list)  # 隣接リスト: {ノード: [(隣接ノード, 重み), ...]}
        self.depth = [0] * n  # 各ノードの深さ
        self.dist = [0] * n  # 根からの重み合計
        self.ancestor = [[-1] * self.log for _ in range(n)]  # ダブリングテーブル

    def add_edge(self, u, v, w):
        """辺を追加: uとvを重みwで接続"""
        self.adj[u].append((v, w))
        self.adj[v].append((u, w))

    def dfs(self, u, parent, d, w):
        """DFSで深さ、距離、親を計算"""
        self.depth[u] = d
        self.dist[u] = w
        for v, weight in self.adj[u]:
            if v != parent:
                self.ancestor[v][0] = u
                self.dfs(v, u, d + 1, w + weight)

    def build(self, root=0):
        """ダブリングテーブルの構築"""
        # DFSで初期情報収集
        self.dfs(root, -1, 0, 0)
        # ダブリングテーブルを埋める
        for k in range(1, self.log):
            for u in range(self.n):
                if self.ancestor[u][k - 1] != -1:
                    self.ancestor[u][k] = self.ancestor[self.ancestor[u][k - 1]][k - 1]

    def lca(self, u, v):
        """ノードuとvのLCAを求める"""
        # 深さを揃える
        if self.depth[u] < self.depth[v]:
            u, v = v, u
        for k in range(self.log - 1, -1, -1):
            if (
                self.ancestor[u][k] != -1
                and self.depth[self.ancestor[u][k]] >= self.depth[v]
            ):
                u = self.ancestor[u][k]
        if u == v:
            return u
        # 同時にジャンプ
        for k in range(self.log - 1, -1, -1):
            if self.ancestor[u][k] != self.ancestor[v][k]:
                u = self.ancestor[u][k]
                v = self.ancestor[v][k]
        return self.ancestor[u][0]

    def get_distance(self, u, v):
        """ノードuとvの間の距離（重みの合計）を求める"""
        lca_node = self.lca(u, v)
        return self.dist[u] + self.dist[v] - 2 * self.dist[lca_node]


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
        区間[l,r)のハッシュ値を取得する
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


# グラフ構造
# 無向グラフ
from collections import deque
from typing import List, Tuple


class Graph:
    """
    グラフ構造体
    """

    def __init__(self, N: int, dire: bool = False) -> None:
        """
        Nは頂点数、direは有向グラフかです
        """
        self.N = N
        self.dire = dire
        self.grath = [[] for _ in [0] * self.N]
        self.in_deg = [0] * N

    def new_side(self, a: int, b: int):
        """
        注意　0-indexedが前提
        aとbを辺で繋ぎます
        有向グラフなら、aからbだけ、無向グラフなら、aからbと、bからaを繋ぎます
        """
        self.grath[a].append(b)
        if self.dire:
            self.in_deg[b] += 1

        if not self.dire:
            self.grath[b].append(a)

    def side_input(self):
        """
        標準入力で、新しい辺を追加します
        """
        a, b = map(lambda x: int(x) - 1, input().split())
        self.new_side(a, b)

    def input(self, M: int):
        """
        標準入力で複数行受け取り、各行の内容で辺を繋ぎます
        """
        for _ in [0] * M:
            self.side_input()

    def get(self, a: int):
        """
        頂点aの隣接頂点を出力します
        """
        return self.grath[a]

    def all(self) -> List[List[int]]:
        """
        グラフの隣接リストをすべて出力します
        """
        return self.grath

    def topological(self, unique: bool = False) -> List[int]:
        """
        トポロジカルソートします
        有向グラフ限定です

        引数のuniqueは、トポロジカルソート結果が、一意に定まらないとエラーを吐きます
        閉路がある、または、uniqueがTrueで一意に定まらなかった時は、[-1]を返します
        """
        if not self.dire:
            raise ValueError("グラフが有向グラフでは有りません (╥﹏╥)")

        in_deg = self.in_deg[:]

        S: deque[int] = deque([])
        order: List[int] = []

        for i in range(self.N):
            if in_deg[i] == 0:
                S.append(i)

        while S:
            if unique and len(S) != 1:
                return [-1]

            cur = S.pop()
            order.append(cur)

            for nxt in self.get(cur):
                in_deg[nxt] -= 1

                if in_deg[nxt] == 0:
                    S.append(nxt)

        if sum(in_deg) > 0:
            return [-1]
        else:
            return [x for x in order]


class GraphW:
    """
    重み付きグラフ
    """

    def __init__(self, N: int, dire: bool = False) -> None:
        self.N = N
        self.dire = dire
        self.grath = [[] for _ in [0] * self.N]

    def new_side(self, a: int, b: int, w: int):
        """
        注意　0-indexedが前提
        aとbを辺で繋ぎます
        有向グラフなら、aからbだけ、無向グラフなら、aからbと、bからaを繋ぎます
        """
        self.grath[a].append((b, w))
        if not self.dire:
            self.grath[b].append((a, w))

    def side_input(self):
        """
        標準入力で、新しい辺を追加します
        """
        a, b, w = map(lambda x: int(x) - 1, input().split())
        self.new_side(a, b, w + 1)

    def input(self, M: int):
        """
        標準入力で複数行受け取り、各行の内容で辺を繋ぎます
        """
        for _ in [0] * M:
            self.side_input()

    def get(self, a: int) -> List[Tuple[int]]:
        """
        頂点aの隣接頂点を出力します
        """
        return self.grath[a]

    def all(self) -> List[List[Tuple[int]]]:
        """
        グラフの隣接リストをすべて出力します
        """
        return self.grath


from collections import defaultdict
from typing import List


# UnionFind木
class UnionFind:
    """
    rollbackをデフォルトで装備済み
    計算量は、経路圧縮を行わないため、基本的なUnionFindの動作は、一回あたり、O(log N)
    rollbackは、一回あたり、O(1)で行える。
    """

    def __init__(self, n: int) -> None:
        self.size = n
        self.data = [-1] * n
        self.hist = []

    def root(self, vtx: int) -> int:
        """
        頂点vtxの親を出力します
        """
        if self.data[vtx] < 0:
            return vtx

        return self.root(self.data[vtx])

    def same(self, a: int, b: int):
        """
        aとbが連結しているかどうか判定します
        """
        return self.root(a) == self.root(b)

    def unite(self, a: int, b: int) -> bool:
        """
        aとbを結合します
        rootが同じでも、履歴には追加します
        """
        ra, rb = self.root(a), self.root(b)

        # 履歴を作成する
        new_hist = [ra, rb, self.data[ra], self.data[rb]]
        self.hist.append(new_hist)

        if ra == rb:
            return False

        if self.data[ra] > self.data[rb]:
            ra, rb = rb, ra

        self.data[ra] += self.data[rb]
        self.data[rb] = ra

        return True

    def rollback(self):
        """
        undoします
        redoはありません
        """
        if not self.hist:
            return False

        ra, rb, da, db = self.hist.pop()
        self.data[ra] = da
        self.data[rb] = db
        return True

    def all(self) -> List[List[int]]:
        D = defaultdict(list)

        for i in range(self.size):
            D[self.root(i)].append(i)

        res = []

        for l in D.values():
            res.append(l)

        return res


from typing import List


class PotentialUnionFind:
    def __init__(self, n: int) -> None:
        """
        重み付きunionfind
        俗に言う、牛ゲー

        uniteは、差を指定して、uniteします
        """
        self.data: List[int] = [-1] * n
        self.pot: List[int] = [0] * n

    def root(self, vtx: int) -> int:
        """
        頂点vtxの親を出力します
        ポテンシャルは出力しません
        """
        if self.data[vtx] < 0:
            return vtx

        rt = self.root(self.data[vtx])
        self.pot[vtx] += self.pot[self.data[vtx]]
        self.data[vtx] = rt

        return rt

    def potential(self, vtx: int) -> int:
        """
        頂点vtxのポテンシャルを出力します
        """
        self.root(vtx)

        return self.pot[vtx]

    def same(self, a: int, b: int) -> bool:
        """
        頂点aと頂点bが同じ連結成分かを判定します
        """
        return self.root(a) == self.root(b)

    def unite(self, a: int, b: int, p: int) -> bool:
        """
        頂点aから頂点bを、pの距離でmergeします
        計算量はlog nです
        """
        p += self.potential(b) - self.potential(a)
        a, b = self.root(a), self.root(b)

        if a == b:
            return False

        if self.data[a] < self.data[b]:
            a, b = b, a
            p *= -1  # ポテンシャルもswapします

        self.data[b] += self.data[a]
        self.data[a] = b
        self.pot[a] = p

        return True

    def diff(self, a: int, b: int) -> int:
        """
        頂点aから頂点bの距離を、出力します
        """

        return self.potential(a) - self.potential(b)


from typing import Any, Callable, List, Tuple


def _keys_for_heapq(x: Any):
    """
    先頭の値を取得する
    """

    cur = x

    while True:
        try:
            cur = cur[0]
        except TypeError:
            break

    return cur


class HeapBase:
    def __init__(
        self, arr: List[Any] = [], key: Callable[Any, Any] = _keys_for_heapq
    ) -> None:
        """
        arrはソート済みが前提です
        """
        self.key: Callable[Any, Any] = key
        self.lis: List[Tuple[Any, Any]] = [(self.key(x), x) for x in arr]

    def _op(self, a: int, b: int) -> bool:
        # aが親 bが子って感じだよ
        assert 0 <= a < b < len(self.lis)
        return True

    def push(self, x: Any) -> None:
        self.lis.append((self.key(x), x))
        i = len(self.lis) - 1
        while i != 0:
            p = (i - 1) // 2
            if self._op(p, i):
                self.lis[i], self.lis[p] = self.lis[p], self.lis[i]
                i = p
            else:
                break

    def pop(self) -> Any:
        assert len(self.lis) > 0
        res = self.lis[0][1]  # Return the original value (not the key)
        self.lis[0] = self.lis[-1]  # Move the last element to the root
        self.lis.pop()  # Remove the last element

        if not self.lis:  # If the heap is empty, return early
            return res

        # Restore heap property by sifting down
        i = 0
        while i * 2 + 1 < len(self.lis):  # While there is at least one child
            c1 = i * 2 + 1  # Left child
            c2 = i * 2 + 2  # Right child

            # Pick the smaller of the two children (if right child exists)
            smallest = c1
            if c2 < len(self.lis) and self._op(c1, c2):
                smallest = c2

            # If the parent is larger than the smallest child, swap
            if self._op(i, smallest):
                self.lis[i], self.lis[smallest] = self.lis[smallest], self.lis[i]
                i = smallest
            else:
                break

        return res

    def __len__(self) -> int:
        return len(self.lis)

    def __getitem__(self, i: int):
        return self.lis[i][1]


class HeapMin(HeapBase):
    def _op(self, a: int, b: int) -> bool:
        return self.lis[a][0] > self.lis[b][0]


class HeapMax(HeapBase):
    def _op(self, a: int, b: int) -> bool:
        return self.lis[a][0] < self.lis[b][0]


# Trie木
class Trie:
    class Data:
        def __init__(self, value, ind):
            self.count = 1
            self.value = value
            self.childs = {}
            self.ind = ind

    def __init__(self):
        self.data = [self.Data("ab", 0)]  # 初期値はabにして被らないようにする

    def add(self, value: str) -> int:
        cur = 0
        result = 0

        # 再帰的に探索する
        for t in value:
            childs = self.data[cur].childs  # 参照渡しで

            if t in childs:
                self.data[childs[t]].count += 1
            else:
                nd = self.Data(t, len(self.data))
                childs[t] = len(self.data)
                self.data.append(nd)

            result += self.data[childs[t]].count - 1
            cur = childs[t]

        return result

    def lcp_max(self, value: str) -> int:
        cur = 0
        result = 0

        for t in value:
            childs = self.data[cur].childs

            if t not in childs:
                break

            if self.data[childs[t]].count == 1:
                break

            cur = childs[t]
            result += 1

        return result

    def lcp_sum(self, value: str) -> int:
        cur = 0
        result = 0

        for t in value:
            childs = self.data[cur].childs

            if t not in childs:
                break

            if self.data[childs[t]].count == 1:
                break

            cur = childs[t]
            result += self.data[childs[t]].count - 1

        return result


import math
from typing import Any, Callable, List


def mo_algorithm(
    N: int,
    queries: List[Any],
    add: Callable[[int], Any],
    delete: Callable[[int], Any],
    getvalue: Callable[[], Any],
) -> List[Any]:
    """
    Mo's algorithmの関数
    queriesは、(左端, 右端)で1-indexed
    addはあるindexが追加される時の値を現在の値にする
    deleteはあるindexが削除される時の値を現在の値にする
    getvalueは現在の値を返す
    """
    Q = len(queries)
    res = [None] * Q
    M = int(max(1, 1.0 * N / max(1, math.sqrt(Q * 2.0 / 3.0))))

    queries = [(l, r, i) for i, (l, r) in enumerate(queries)]
    queries.sort(key=lambda x: (x[0] // M, x[1] if (x[0] // M) % 2 == 0 else -x[1]))

    cl, cr = 0, -1

    for l, r, ind in queries:
        l -= 1
        r -= 1
        while cl > l:
            cl -= 1
            add(cl)

        while cr < r:
            cr += 1
            add(cr)

        while cl < l:
            delete(cl)
            cl += 1

        while cr > r:
            delete(cr)
            cr -= 1

        res[ind] = getvalue()

    return res


import math
from typing import Any, Callable, List


class SquareDivision:
    def __init__(self, lis: List[Any], op: Callable[[Any, Any], Any]) -> None:
        self.n = len(lis)
        self.op = op
        self.block_size = math.isqrt(self.n)
        self.blocks = []
        self.lis = lis[:]

        for i in range(0, self.n, self.block_size):
            block_val = lis[i]
            for k in range(i + 1, min(i + self.block_size, self.n)):
                block_val = self.op(block_val, lis[k])
            self.blocks.append(block_val)

        self.m = len(self.blocks)

    def get_block_index_left(self, i: int) -> int:
        return i // self.block_size

    def get_block_index_right(self, i: int) -> int:
        return (i + self.block_size - 1) // self.block_size

    def prod(self, l: int, r: int) -> Any:
        """
        rは0-indexedなのに注意してください
        """
        assert 0 <= l <= r < self.n

        l_block_left = self.get_block_index_left(l)
        r_block_left = self.get_block_index_left(r)

        if l_block_left == r_block_left:
            res = self.lis[l]
            for k in range(l + 1, r + 1):
                res = self.op(res, self.lis[k])
            return res

        res = self.lis[l]
        for i in range(l + 1, min((l_block_left + 1) * self.block_size, self.n)):
            res = self.op(res, self.lis[i])

        for block_ind in range(l_block_left + 1, r_block_left):
            res = self.op(res, self.blocks[block_ind])

        for i in range(r_block_left * self.block_size, r + 1):
            res = self.op(res, self.lis[i])

        return res

    def update(self, i: int, x: Any) -> None:
        assert 0 <= i < self.n
        self.lis[i] = x
        block_ind = self.get_block_index_left(i)
        start = block_ind * self.block_size
        end = min(start + self.block_size, self.n)
        if start < self.n:
            self.blocks[block_ind] = self.lis[start]
            for j in range(start + 1, end):
                self.blocks[block_ind] = self.op(self.blocks[block_ind], self.lis[j])

    def get(self, i: int) -> Any:
        assert 0 <= i < self.n
        return self.lis[i]


class SquareDivisionSpeedy(SquareDivision):
    def __init__(
        self,
        lis: List[Any],
        op: Callable[[Any, Any], Any],
        delete: Callable[[Any, Any], Any],
    ) -> None:
        self.delete = delete
        super().__init__(lis, op)

    def update(self, i: int, x: Any) -> None:
        assert 0 <= i < self.n

        block_ind = self.get_block_index_left(i)
        self.blocks[block_ind] = self.delete(self.blocks[block_ind], self.lis[i])
        self.lis[i] = x
        self.blocks[block_ind] = self.op(self.blocks[block_ind], self.lis[i])


from typing import List


class BIT:
    """
    BITです
    要素更新と、区間和を求める事ができます
    1-indexedです

    計算量は、一回の動作につきすべてO(log n)です
    """

    def __init__(self, n: int) -> None:
        self.n: int = n
        self.bit: List[int] = [0] * (n + 1)

    def sum(self, i: int) -> int:
        """
        i番目までの和を求めます
        計算量は、O(log n)です
        """
        res = 0

        while i:
            res += self.bit[i]
            i -= -i & i

        return res

    def interval_sum(self, l: int, r: int) -> int:
        """
        lからrまでの総和を求められます
        lは0-indexedで、rは1-indexedにしてください
        """
        return self.sum(r) - self.sum(l)

    def add(self, i: int, x: int):
        """
        i番目の要素にxを足します
        計算量は、O(log n)です
        """
        if i == 0:
            raise IndexError("このデータ構造は、1-indexedです")

        while i <= self.n:
            self.bit[i] += x
            i += -i & i


from typing import Any, Callable


class DualSegmentTree:
    def __init__(self, op: Callable[[Any, Any], Any], e: Any, n: int) -> None:
        """
        区間作用/一点取得のセグメント木
        opは区間作用用の関数
        eは初期値
        vは長さ
        """
        self._op: Callable[[Any, Any], Any] = op
        self._e: Any = e
        self._n: int = n
        self.n: int = 1 << (n - 1).bit_length()
        self.data = [e] * (self.n * 2)

    def apply(self, l, r, x) -> None:
        """
        区間[l,r)にxを適用
        """
        assert 0 <= l <= r <= self.n
        l += self.n
        r += self.n

        while l < r:
            if l & 1:
                self.data[l] = self._op(self.data[l], x)
                l += 1

            if r & 1:
                self.data[r - 1] = self._op(self.data[r - 1], x)

            l >>= 1
            r >>= 1

    def get(self, p: int) -> Any:
        """
        pの値を取得する
        """
        assert 0 <= p < self.n

        res = self._e
        p += self.n

        while p:
            res = self._op(res, self.data[p])
            p >>= 1

        return res


from typing import Tuple


def euclid_dis(x1: int, y1: int, x2: int, y2: int) -> int:
    """
    ユークリッド距離を計算します

    注意:
    この関数はsqrtを取りません(主に少数誤差用)
    sqrtを取りたい場合は、自分で計算してください
    """

    return ((x1 - x2) ** 2) + ((y1 - y2) ** 2)


def manhattan_dis(x1: int, y1: int, x2: int, y2: int) -> int:
    """
    マンハッタン距離を計算します
    """

    return abs(x1 - x2) + abs(y1 - y2)


def manhattan_45turn(x: int, y: int) -> Tuple[int]:
    """
    座標を45度回転します
    回転すると、マンハッタン距離が、チェビシェフ距離になるので、距離の最大値などが簡単に求められます
    """

    res_x = x - y
    res_y = x + y

    return res_x, res_y


def chebyshev_dis(x1: int, y1: int, x2: int, y2: int) -> int:
    """
    チェビシェフ距離を計算します
    """

    return max(abs(x1 - x2), abs(y1 - y2))


# 便利変数
INF = 1 << 63
lowerlist = list("abcdefghijklmnopqrstuvwxyz")
upperlist = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
MOVES1 = [(0, 1), (0, -1), (1, 0), (-1, 0)]
MOVES2 = MOVES1 + [(1, 1), (1, -1), (-1, 1), (-1, -1)]

# コード
