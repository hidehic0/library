---
data:
  _extendedDependsOn: []
  _extendedRequiredBy: []
  _extendedVerifiedWith: []
  _isVerificationFailed: false
  _pathExtension: py
  _verificationStatusIcon: ':warning:'
  attributes:
    links: []
  bundledCode: "Traceback (most recent call last):\n  File \"/opt/hostedtoolcache/Python/3.13.2/x64/lib/python3.13/site-packages/onlinejudge_verify/documentation/build.py\"\
    , line 71, in _render_source_code_stat\n    bundled_code = language.bundle(stat.path,\
    \ basedir=basedir, options={'include_paths': [basedir]}).decode()\n          \
    \         ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n\
    \  File \"/opt/hostedtoolcache/Python/3.13.2/x64/lib/python3.13/site-packages/onlinejudge_verify/languages/python.py\"\
    , line 96, in bundle\n    raise NotImplementedError\nNotImplementedError\n"
  code: "r\"\"\"\n ______________________\n< it's hidehico's code >\n ----------------------\n\
    \   \\\n    \\\n        .--.\n       |o_o |\n       |:_/ |\n      //   \\ \\\n\
    \     (|     | )\n    /'\\_   _/`\\\n    \\___)=(___/\n\"\"\"\n\n# \u30E9\u30A4\
    \u30D6\u30E9\u30EA\u3068\u95A2\u6570\u3068\u4FBF\u5229\u5909\u6570\n# \u30E9\u30A4\
    \u30D6\u30E9\u30EA\nimport bisect\nimport heapq\nimport sys\nimport unittest\n\
    from collections import Counter, defaultdict, deque\nfrom itertools import permutations\n\
    from math import gcd, lcm, pi\nfrom typing import Any, List\n\n# from atcoder.segtree\
    \ import SegTree\n# from atcoder.lazysegtree import LazySegTree\n# from atcoder.dsu\
    \ import DSU\n\n# cortedcontainers\u306F\u4F7F\u3046\u3068\u304D\u3060\u3051 wandbox\u975E\
    \u5BFE\u5FDC\u306A\u306E\u3067\n# from sortedcontainers import SortedDict, SortedSet,\
    \ SortedList\n\n# import pypyjit\n# pypyjit.set_param(\"max_unroll_recursion=-1\"\
    )\n\nsys.setrecursionlimit(5 * 10**5)\nfrom typing import List\n\n\n# \u6570\u5B66\
    \u578B\u95A2\u6570\ndef is_prime(n: int) -> int:\n    \"\"\"\n    \u7D20\u6570\
    \u5224\u5B9A\u3057\u307E\u3059\n    \u8A08\u7B97\u91CF\u306F\u5B9A\u6570\u6642\
    \u9593\u3067\u3059\u3002\u6B63\u78BA\u306B\u306F\u3001\u7E70\u308A\u8FD4\u3057\
    \u4E8C\u4E57\u6CD5\u306E\u8A08\u7B97\u91CF\u306B\u3088\u308A\u3067\u3059\n   \
    \ \u30A2\u30EB\u30B4\u30EA\u30BA\u30E0\u306F\u30DF\u30E9\u30FC\u30E9\u30D3\u30F3\
    \u306E\u7D20\u6570\u5224\u5B9A\u3092\u4F7F\u7528\u3057\u3066\u3044\u307E\u3059\
    \n    n\u304C2^64\u3092\u8D8A\u3048\u308B\u3068\u52D5\u4F5C\u3057\u307E\u305B\u3093\
    \n    \"\"\"\n    if n == 1:\n        return False\n\n    def f(a, t, n):\n  \
    \      x = pow(a, t, n)\n        nt = n - 1\n        while t != nt and x != 1\
    \ and x != nt:\n            x = pow(x, 2, n)\n            t <<= 1\n\n        return\
    \ t & 1 or x == nt\n\n    if n == 2:\n        return True\n    elif n % 2 == 0:\n\
    \        return False\n\n    d = n - 1\n    d >>= 1\n\n    while d & 1 == 0:\n\
    \        d >>= 1\n\n    checklist = (\n        [2, 7, 61] if 2**32 > n else [2,\
    \ 325, 9375, 28178, 450775, 9780504, 1795265022]\n    )\n\n    for i in checklist:\n\
    \        if i >= n:\n            break\n        if not f(i, d, n):\n         \
    \   return False\n\n    return True\n\n\ndef eratosthenes(n: int) -> List[int]:\n\
    \    \"\"\"\n    n\u4EE5\u4E0B\u306E\u7D20\u6570\u3092\u5217\u6319\u3057\u307E\
    \u3059\n    \u8A08\u7B97\u91CF\u306F\u3001O(n log log n)\u3067\u3059\n    \u5148\
    \u7A0B\u306E\u7D20\u6570\u5224\u5B9A\u6CD5\u3067\u5217\u6319\u3059\u308B\u3088\
    \u308A\u3082\u3001\u5C11\u3057\u901F\u3044\u3067\u3059\n    \u5217\u6319\u3057\
    \u305F\u7D20\u6570\u306F\u6607\u9806\u306B\u4E26\u3093\u3067\u3044\u307E\u3059\
    \n    \u30A2\u30EB\u30B4\u30EA\u30BA\u30E0\u306F\u30A8\u30E9\u30C8\u30B9\u30C6\
    \u30CD\u30B9\u3067\u3059\n    \"\"\"\n    primes = [True] * (n + 1)\n    primes[0],\
    \ primes[1] = False, False\n    i = 2\n    while i**2 <= n:\n        if primes[i]:\n\
    \            for k in range(i * 2, n + 1, i):\n                primes[k] = False\n\
    \n        i += 1\n\n    return [i for i, p in enumerate(primes) if p]\n\n\ndef\
    \ calc_divisors(n: int):\n    \"\"\"\n    N\u306E\u7D04\u6570\u5217\u6319\u3057\
    \u307E\u3059\n    \u8A08\u7B97\u91CF\u306F\u3001\u221AN\u3067\u3059\n    \u7D04\
    \u6570\u306F\u6607\u9806\u306B\u4E26\u3093\u3067\u3044\u307E\u3059\n    \"\"\"\
    \n    result = []\n\n    for i in range(1, n + 1):\n        if i * i > n:\n  \
    \          break\n\n        if n % i != 0:\n            continue\n\n        result.append(i)\n\
    \        if n // i != i:\n            result.append(n // i)\n\n    return sorted(result)\n\
    \n\ndef factorization(n: int) -> List[List[int]]:\n    \"\"\"\n    n\u3092\u7D20\
    \u56E0\u6570\u5206\u89E3\u3057\u307E\u3059\n    \u8A08\u7B97\u91CF\u306F\u3001\
    \u221AN\u3067\u3059(\u8981\u6539\u5584)\n    \u8907\u6570\u56DE\u7D20\u56E0\u6570\
    \u5206\u89E3\u3092\u884C\u306A\u3046\u5834\u5408\u306F\u3001\u221AN\u4EE5\u4E0B\
    \u306E\u7D20\u6570\u3092\u5217\u6319\u3057\u305F\u306E\u3067\u8A66\u3057\u5272\
    \u308A\u3057\u305F\u6CD5\u304C\u901F\u3044\u3067\u3059\n    \"\"\"\n    result\
    \ = []\n    tmp = n\n    for i in range(2, int(-(-(n**0.5) // 1)) + 1):\n    \
    \    if tmp % i == 0:\n            cnt = 0\n            while tmp % i == 0:\n\
    \                cnt += 1\n                tmp //= i\n            result.append([i,\
    \ cnt])\n\n    if tmp != 1:\n        result.append([tmp, 1])\n\n    if result\
    \ == []:\n        result.append([n, 1])\n\n    return result\n\n\ndef factorization_plural(L:\
    \ List[int]) -> List[List[List[int]]]:\n    \"\"\"\n    \u8907\u6570\u306E\u6570\
    \u306E\u7D20\u56E0\u6570\u5206\u89E3\u3092\u884C\u306A\u3044\u307E\u3059\n   \
    \ \u8A08\u7B97\u91CF\u306F\u3001O(N * (\u221Amax(L) log log \u221Amax(L)))\n \
    \   \u307F\u305F\u3044\u306A\u611F\u3058\u3067\u3059\n\n    \u6700\u521D\u306B\
    \u7D20\u6570\u3092\u5217\u6319\u3059\u308B\u305F\u3081\u3001\u666E\u901A\u306E\
    \u7D20\u56E0\u6570\u5206\u89E3\u3088\u308A\u52B9\u7387\u304C\u3044\u3044\u3067\
    \u3059\n    \"\"\"\n    res = []\n    primes = eratosthenes(int(max(L) ** 0.5)\
    \ + 20)\n\n    def solve(n):\n        t = []\n        for p in primes:\n     \
    \       if n % p == 0:\n                cnt = 0\n                while n % p ==\
    \ 0:\n                    cnt += 1\n                    n //= p\n\n          \
    \      t.append([p, cnt])\n\n        if n != 1:\n            t.append([n, 1])\n\
    \n        if t == []:\n            t.append([n, 1])\n\n        return t\n\n  \
    \  for n in L:\n        res.append(solve(n))\n\n    return res\n\n\ndef simple_sigma(n:\
    \ int) -> int:\n    \"\"\"\n    1\u304B\u3089n\u307E\u3067\u306E\u7DCF\u548C\u3092\
    \u6C42\u3081\u308B\u95A2\u6570\n    \u3064\u307E\u308A\u548C\u306E\u516C\u5F0F\
    \n    \"\"\"\n    return (n * (n + 1)) // 2\n\n\ndef comb(n: int, r: int, mod:\
    \ int | None = None) -> int:\n    \"\"\"\n    \u9AD8\u901F\u306A\u306F\u305A\u306E\
    \u4E8C\u9805\u4FC2\u6570\n    mod\u3092\u6307\u5B9A\u3059\u308C\u3070\u3001mod\u4ED8\
    \u304D\u306B\u306A\u308B\n    \"\"\"\n    a = 1\n\n    for i in range(n - r +\
    \ 1, n + 1):\n        a *= i\n\n        if mod:\n            a %= mod\n\n    b\
    \ = 1\n\n    for i in range(1, r + 1):\n        b *= i\n        if mod:\n    \
    \        b %= mod\n\n    if mod:\n        return a * pow(b, -1, mod) % mod\n \
    \   else:\n        return a * b\n\n\n# \u591A\u6B21\u5143\u914D\u5217\u4F5C\u6210\
    \nfrom typing import List, Any\n\n\ndef create_array1(n: int, default: Any = 0)\
    \ -> List[Any]:\n    \"\"\"\n    1\u6B21\u5143\u914D\u5217\u3092\u521D\u671F\u5316\
    \u3059\u308B\u95A2\u6570\n    \"\"\"\n    return [default] * n\n\n\ndef create_array2(a:\
    \ int, b: int, default: Any = 0) -> List[List[Any]]:\n    \"\"\"\n    2\u6B21\u5143\
    \u914D\u5217\u3092\u521D\u671F\u5316\u3059\u308B\u95A2\u6570\n    \"\"\"\n   \
    \ return [[default] * b for _ in [0] * a]\n\n\ndef create_array3(a: int, b: int,\
    \ c: int, default: Any = 0) -> List[List[List[Any]]]:\n    \"\"\"\n    3\u6B21\
    \u5143\u914D\u5217\u3092\u521D\u671F\u5316\u3059\u308B\u95A2\u6570\n    \"\"\"\
    \n    return [[[default] * c for _ in [0] * b] for _ in [0] * a]\n\n\nfrom typing\
    \ import Callable\n\n\ndef binary_search(\n    fn: Callable[[int], bool], right:\
    \ int = 0, left: int = -1, return_left: bool = True\n) -> int:\n    \"\"\"\n \
    \   \u4E8C\u5206\u63A2\u7D22\u306E\u62BD\u8C61\u7684\u306A\u30E9\u30A4\u30D6\u30E9\
    \u30EA\n    \u8A55\u4FA1\u95A2\u6570\u306E\u7D50\u679C\u306B\u5FDC\u3058\u3066\
    \u3001\u4E8C\u5206\u63A2\u7D22\u3059\u308B\n    \u6700\u7D42\u7684\u306B\u306F\
    left\u3092\u51FA\u529B\u3057\u307E\u3059\n\n    \u95A2\u6570\u306E\u30C6\u30F3\
    \u30D7\u30EC\u30FC\u30C8\n    def check(mid:int):\n        if A[mid] > x:\n  \
    \          return True\n        else:\n            return False\n\n    mid\u306F\
    \u5FC5\u9808\u3067\u3059\u3002\u305D\u308C\u4EE5\u5916\u306F\u3054\u81EA\u7531\
    \u306B\u3069\u3046\u305E\n    \"\"\"\n    while right - left > 1:\n        mid\
    \ = (left + right) // 2\n\n        if fn(mid):\n            left = mid\n     \
    \   else:\n            right = mid\n\n    return left if return_left else right\n\
    \n\ndef mod_add(a: int, b: int, mod: int):\n    \"\"\"\n    \u8DB3\u3057\u7B97\
    \u3057\u3066mod\u3092\u53D6\u3063\u305F\u5024\u3092\u51FA\u529B\n    O(1)\n  \
    \  \"\"\"\n    return (a + b) % mod\n\n\ndef mod_sub(a: int, b: int, mod: int):\n\
    \    \"\"\"\n    \u5F15\u304D\u7B97\u3057\u3066mod\u3092\u53D6\u3063\u305F\u5024\
    \u3092\u51FA\u529B\n    O(1)\n    \"\"\"\n    return (a - b) % mod\n\n\ndef mod_mul(a:\
    \ int, b: int, mod: int):\n    \"\"\"\n    \u639B\u3051\u7B97\u3057\u3066mod\u3092\
    \u53D6\u3063\u305F\u5024\u3092\u51FA\u529B\n    O(1)\n    \"\"\"\n    return (a\
    \ * b) % mod\n\n\ndef mod_div(a: int, b: int, mod: int):\n    \"\"\"\n    \u5272\
    \u308A\u7B97\u3057\u3066mod\u3092\u53D6\u3063\u305F\u5024\u3092\u51FA\u529B\n\
    \    \u30D5\u30A7\u30EB\u30DE\u30FC\u306E\u5C0F\u5B9A\u7406\u3092\u4F7F\u3063\u3066\
    \u8A08\u7B97\u3057\u307E\u3059\n    O(log mod)\n    \"\"\"\n    return (a * pow(b,\
    \ mod - 2, mod)) % mod\n\n\nclass ModInt:\n    def __init__(self, x: int, mod:\
    \ int = 998244353) -> None:\n        self.x = x % mod\n        self.mod = mod\n\
    \n    def val(self):\n        return self.x\n\n    def rhs(self, rhs) -> int:\n\
    \        return rhs.x if isinstance(rhs, ModInt) else rhs\n\n    def __add__(self,\
    \ rhs) -> int:\n        return mod_add(self.x, self.rhs(rhs), self.mod)\n\n  \
    \  def __iadd__(self, rhs) -> \"ModInt\":\n        self.x = self.__add__(rhs)\n\
    \n        return self\n\n    def __sub__(self, rhs) -> int:\n        return mod_sub(self.x,\
    \ self.rhs(rhs), self.mod)\n\n    def __isub__(self, rhs) -> \"ModInt\":\n   \
    \     self.x = self.__sub__(rhs)\n\n        return self\n\n    def __mul__(self,\
    \ rhs):\n        return mod_mul(self.x, self.rhs(rhs), self.mod)\n\n    def __imul__(self,\
    \ rhs):\n        self.x = self.__mul__(rhs)\n\n        return self\n\n    def\
    \ __truediv__(self, rhs):\n        return mod_div(self.x, self.rhs(rhs), self.mod)\n\
    \n    def __itruediv__(self, rhs):\n        self.x = self.__truediv__(rhs)\n\n\
    \        return self\n\n    def __floordiv__(self, rhs):\n        return (self.x\
    \ // self.rhs(rhs)) % self.mod\n\n    def __ifloordiv__(self, rhs):\n        self.x\
    \ = self.__floordiv__(rhs)\n\n        return self\n\n    def __pow__(self, rhs):\n\
    \        return pow(self.x, self.rhs(rhs), self.mod)\n\n    def __eq__(self, rhs)\
    \ -> bool:\n        return self.rhs(rhs) == self.x\n\n    def __ne__(self, rhs)\
    \ -> bool:\n        return self.rhs(rhs) != self.x\n\n\n# \u6A19\u6E96\u5165\u529B\
    \u95A2\u6570\nimport sys\nfrom typing import Any, List\n\n\ndef s() -> str:\n\
    \    \"\"\"\n    \u4E00\u884C\u306B\u4E00\u3064\u306Estring\u3092input\n    \"\
    \"\"\n    return sys.stdin.readline().rstrip()\n\n\ndef sl() -> List[str]:\n \
    \   \"\"\"\n    \u4E00\u884C\u306B\u8907\u6570\u306Estring\u3092input\n    \"\"\
    \"\n    return s().split()\n\n\ndef ii() -> int:\n    \"\"\"\n    \u4E00\u3064\
    \u306Eint\n    \"\"\"\n    return int(s())\n\n\ndef il(add_num: int = 0) -> List[int]:\n\
    \    \"\"\"\n    \u4E00\u884C\u306B\u8907\u6570\u306Eint\n    \"\"\"\n    return\
    \ list(map(lambda i: int(i) + add_num, sl()))\n\n\ndef li(n: int, func, *args)\
    \ -> List[List[Any]]:\n    \"\"\"\n    \u8907\u6570\u884C\u306E\u5165\u529B\u3092\
    \u30B5\u30DD\u30FC\u30C8\n    \"\"\"\n    return [func(*args) for _ in [0] * n]\n\
    \n\n# YesNo\u95A2\u6570\ndef YesNoTemplate(state: bool, upper: bool = False) ->\
    \ str:\n    \"\"\"\n    state\u304CTrue\u306A\u3089\u3001upper\u306B\u5FDC\u3058\
    \u3066Yes,YES\u3092return\n    state\u304CFalse\u306A\u3089\u3001upper\u306B\u5FDC\
    \u3058\u3066No,NO\u3092return\u3059\u308B\n    \"\"\"\n    YES = [\"Yes\", \"\
    YES\"]\n    NO = [\"No\", \"NO\"]\n\n    if state:\n        return YES[int(upper)]\n\
    \    else:\n        return NO[int(upper)]\n\n\ndef YN(state: bool, upper: bool\
    \ = False) -> None:\n    \"\"\"\n    \u5148\u7A0B\u306EYesNoTemplate\u95A2\u6570\
    \u306E\u7D50\u679C\u3092\u51FA\u529B\u3059\u308B\n    \"\"\"\n    res = YesNoTemplate(state,\
    \ upper)\n\n    print(res)\n\n\ndef YE(state: bool, upper: bool = False) -> bool\
    \ | None:\n    \"\"\"\n    bool\u304CTrue\u306A\u3089Yes\u3092\u51FA\u529B\u3057\
    \u3066exit\n    \"\"\"\n\n    if not state:\n        return False\n\n    YN(True,\
    \ upper)\n    exit()\n\n\ndef NE(state: bool, upper: bool = False) -> bool | None:\n\
    \    \"\"\"\n    bool\u304CTrue\u306A\u3089No\u3092\u51FA\u529B\u3057\u3066exit\n\
    \    \"\"\"\n\n    if not state:\n        return False\n\n    YN(False, upper)\n\
    \    exit()\n\n\ndef coordinate_check(x: int, y: int, H: int, W: int) -> bool:\n\
    \    \"\"\"\n    \u5EA7\u6A19\u304C\u30B0\u30EA\u30C3\u30C9\u306E\u7BC4\u56F2\u5185\
    \u306B\u3042\u308B\u304B\u30C1\u30A7\u30C3\u30AF\u3059\u308B\u95A2\u6570\n   \
    \ 0-indexed\u304C\u524D\u63D0\n    \"\"\"\n\n    return 0 <= x < H and 0 <= y\
    \ < W\n\n\nfrom typing import List, Tuple\n\n\ndef grid_moves(\n    x: int,\n\
    \    y: int,\n    H: int,\n    W: int,\n    moves: List[Tuple[int]] = [(0, 1),\
    \ (0, -1), (1, 0), (-1, 0)],\n    *check_funcs,\n) -> List[Tuple[int]]:\n    \"\
    \"\"\n    \u73FE\u5728\u306E\u5EA7\u6A19\u304B\u3089\u3001\u79FB\u52D5\u53EF\u80FD\
    \u306A\u5EA7\u6A19\u3092moves\u3092\u3082\u3068\u306B\u5217\u6319\u3057\u307E\u3059\
    \u3002\n    x\u3068y\u306F\u73FE\u5728\u306E\u5EA7\u6A19\n    H\u3068W\u306F\u30B0\
    \u30EA\u30C3\u30C9\u306E\u30B5\u30A4\u30BA\n    moves\u306F\u79FB\u52D5\u3059\u308B\
    \u5EA7\u6A19\u304C\u3044\u304F\u3064\u304B\u3092\u4FDD\u5B58\u3059\u308B\n   \
    \ check_funcs\u306F\u3001\u305D\u306E\u5EA7\u6A19\u306E\u70B9\u304C#\u3060\u3068\
    \u304B\u3092\u81EA\u524D\u3067\u5B9F\u88C5\u3057\u3066\u5224\u5B9A\u306F\u3053\
    \u3061\u3089\u3067\u3059\u308B\u307F\u305F\u3044\u306A\u611F\u3058\n    \u306A\
    \u304Acheck_funcs\u306F\u5F15\u6570\u304Cx\u3068y\u3060\u3051\u3068\u3044\u3046\
    \u306E\u304C\u6761\u4EF6\n    \u8FFD\u52A0\u306E\u5224\u5B9A\u95A2\u6570\u306F\
    \u3001\u5F3E\u304F\u5834\u5408\u306F\u3001False \u305D\u308C\u4EE5\u5916\u306A\
    \u3089True\u3067\n    \"\"\"\n    res = []\n\n    for mx, my in moves:\n     \
    \   nx, ny = x + mx, y + my\n\n        if not coordinate_check(nx, ny, H, W):\n\
    \            continue\n\n        for f in check_funcs:\n            if not f(nx,\
    \ ny):\n                break\n        else:\n            res.append((nx, ny))\n\
    \n    return res\n\n\n# DP\u306E\u30C6\u30F3\u30D7\u30EC\u30FC\u30C8\nfrom typing\
    \ import List\n\n\ndef partial_sum_dp(lis: List[int], X: int) -> List[bool]:\n\
    \    \"\"\"\n    \u90E8\u5206\u548Cdp\u306E\u30C6\u30F3\u30D7\u30EC\u30FC\u30C8\
    \n    lis\u306F\u54C1\u7269\u3067\u3059\n    dp\u914D\u5217\u306E\u9577\u3055\u306F\
    \u3001X\u306B\u3057\u307E\u3059\n    \u8A08\u7B97\u91CF\u306F\u3001O(X*len(L))\u307F\
    \u305F\u3044\u306A\u611F\u3058\n\n    \u8FD4\u308A\u5024\u306F\u3001dp\u914D\u5217\
    \u3067\u4E2D\u8EAB\u306F\u5230\u9054\u3067\u304D\u305F\u304B\u3092\u3001\u793A\
    \u3059bool\u3067\u3059\n    \"\"\"\n    dp = [False] * (X + 1)\n    dp[0] = True\n\
    \n    for a in lis:\n        for k in reversed(range(len(dp))):\n            if\
    \ not dp[k]:\n                continue\n\n            if k + a >= len(dp):\n \
    \               continue\n\n            dp[k + a] = True\n\n    return dp\n\n\n\
    def knapsack_dp(lis: List[List[int]], W: int) -> List[int]:\n    \"\"\"\n    \u30CA\
    \u30C3\u30D7\u30B5\u30C3\u30AFdp\u306E\u30C6\u30F3\u30D7\u30EC\u30FC\u30C8\n \
    \   lis\u306F\u54C1\u7269\u306E\u30EA\u30B9\u30C8\n    \u539F\u5247\u54C1\u7269\
    \u306F\u3001w,v\u306E\u5F62\u3067\u4E0E\u3048\u3089\u308C\u3001w\u304C\u91CD\u3055\
    \u3001v\u304C\u4FA1\u5024\u3001\u3068\u306A\u308B\n    \u4FA1\u5024\u3068\u91CD\
    \u3055\u3092\u9006\u8EE2\u3055\u305B\u305F\u3044\u5834\u5408\u306F\u81EA\u5206\
    \u3067\u3084\u3063\u3066\u304F\u3060\u3055\u3044\n    dp\u914D\u5217\u306F\u3001\
    \u5B9A\u6570\u500D\u9AD8\u901F\u5316\u306E\u305F\u3081\u3001\u4E00\u6B21\u5143\
    \u914D\u5217\u3068\u3057\u3066\u6271\u3046\n    dp\u914D\u5217\u306E\u9577\u3055\
    \u306F\u3001W\u3068\u3057\u307E\u3059\n    \"\"\"\n\n    dp = [-(1 << 63)] * (W\
    \ + 1)\n    dp[0] = 0\n\n    for w, v in lis:\n        for k in reversed(range(len(dp))):\n\
    \            if w + k >= len(dp):\n                continue\n\n            dp[w\
    \ + k] = max(dp[w + k], dp[k] + v)\n\n    return dp\n\n\ndef article_breakdown(lis:\
    \ List[List[int]]) -> List[List[int]]:\n    \"\"\"\n    \u500B\u6570\u5236\u9650\
    \u4ED8\u304D\u30CA\u30C3\u30D7\u30B5\u30C3\u30AF\u306E\u54C1\u7269\u3092\u5206\
    \u89E3\u3057\u307E\u3059\n    \u500B\u6570\u306E\u5024\u304C\u3001\u5404\u54C1\
    \u7269\u306E\u4E00\u756A\u53F3\u306B\u3042\u308C\u3070\u6B63\u5E38\u306B\u52D5\
    \u4F5C\u3057\u307E\u3059\n    \"\"\"\n    res = []\n    for w, v, c in lis:\n\
    \        k = 1\n        while c > 0:\n            res.append([w * k, v * k])\n\
    \            c -= k\n            k = min(2 * k, c)\n\n    return res\n\n\n# ac_library\u306E\
    \u30E1\u30E2\n\"\"\"\nsegtree\n\n\u521D\u671F\u5316\u3059\u308B\u3068\u304D\n\
    Segtree(op,e,v)\n\nop\u306F\u30DE\u30FC\u30B8\u3059\u308B\u95A2\u6570\n\u4F8B\n\
    \ndef op(a,b):\n    return a+b\n\ne\u306F\u521D\u671F\u5316\u3059\u308B\u5024\n\
    \nv\u306F\u914D\u5217\u306E\u9577\u3055\u307E\u305F\u306F\u3001\u521D\u671F\u5316\
    \u3059\u308B\u5185\u5BB9\n\"\"\"\n# \u30B0\u30E9\u30D5\u69CB\u9020\n# \u7121\u5411\
    \u30B0\u30E9\u30D5\nfrom collections import deque\nfrom typing import List\n\n\
    \nclass Graph:\n    \"\"\"\n    \u30B0\u30E9\u30D5\u69CB\u9020\u4F53\n    \"\"\
    \"\n\n    def __init__(self, N: int, dire: bool = False) -> None:\n        \"\"\
    \"\n        N\u306F\u9802\u70B9\u6570\u3001dire\u306F\u6709\u5411\u30B0\u30E9\u30D5\
    \u304B\u3067\u3059\n        \"\"\"\n        self.N = N\n        self.dire = dire\n\
    \        self.grath = [[] for _ in [0] * self.N]\n        self.in_deg = [0] *\
    \ N\n\n    def new_side(self, a: int, b: int):\n        \"\"\"\n        \u6CE8\
    \u610F\u30000-indexed\u304C\u524D\u63D0\n        a\u3068b\u3092\u8FBA\u3067\u7E4B\
    \u304E\u307E\u3059\n        \u6709\u5411\u30B0\u30E9\u30D5\u306A\u3089\u3001a\u304B\
    \u3089b\u3060\u3051\u3001\u7121\u5411\u30B0\u30E9\u30D5\u306A\u3089\u3001a\u304B\
    \u3089b\u3068\u3001b\u304B\u3089a\u3092\u7E4B\u304E\u307E\u3059\n        \"\"\"\
    \n        self.grath[a].append(b)\n        if self.dire:\n            self.in_deg[b]\
    \ += 1\n\n        if not self.dire:\n            self.grath[b].append(a)\n\n \
    \   def side_input(self):\n        \"\"\"\n        \u6A19\u6E96\u5165\u529B\u3067\
    \u3001\u65B0\u3057\u3044\u8FBA\u3092\u8FFD\u52A0\u3057\u307E\u3059\n        \"\
    \"\"\n        a, b = map(lambda x: int(x) - 1, input().split())\n        self.new_side(a,\
    \ b)\n\n    def input(self, M: int):\n        \"\"\"\n        \u6A19\u6E96\u5165\
    \u529B\u3067\u8907\u6570\u884C\u53D7\u3051\u53D6\u308A\u3001\u5404\u884C\u306E\
    \u5185\u5BB9\u3067\u8FBA\u3092\u7E4B\u304E\u307E\u3059\n        \"\"\"\n     \
    \   for _ in [0] * M:\n            self.side_input()\n\n    def get(self, a: int):\n\
    \        \"\"\"\n        \u9802\u70B9a\u306E\u96A3\u63A5\u9802\u70B9\u3092\u51FA\
    \u529B\u3057\u307E\u3059\n        \"\"\"\n        return self.grath[a]\n\n   \
    \ def all(self) -> List[List[int]]:\n        \"\"\"\n        \u30B0\u30E9\u30D5\
    \u306E\u96A3\u63A5\u30EA\u30B9\u30C8\u3092\u3059\u3079\u3066\u51FA\u529B\u3057\
    \u307E\u3059\n        \"\"\"\n        return self.grath\n\n    def topological(self,\
    \ unique: bool = False) -> List[int]:\n        \"\"\"\n        \u30C8\u30DD\u30ED\
    \u30B8\u30AB\u30EB\u30BD\u30FC\u30C8\u3057\u307E\u3059\n        \u6709\u5411\u30B0\
    \u30E9\u30D5\u9650\u5B9A\u3067\u3059\n\n        \u5F15\u6570\u306Eunique\u306F\
    \u3001\u30C8\u30DD\u30ED\u30B8\u30AB\u30EB\u30BD\u30FC\u30C8\u7D50\u679C\u304C\
    \u3001\u4E00\u610F\u306B\u5B9A\u307E\u3089\u306A\u3044\u3068\u30A8\u30E9\u30FC\
    \u3092\u5410\u304D\u307E\u3059\n        \u9589\u8DEF\u304C\u3042\u308B\u3001\u307E\
    \u305F\u306F\u3001unique\u304CTrue\u3067\u4E00\u610F\u306B\u5B9A\u307E\u3089\u306A\
    \u304B\u3063\u305F\u6642\u306F\u3001[-1]\u3092\u8FD4\u3057\u307E\u3059\n     \
    \   \"\"\"\n        if not self.dire:\n            raise ValueError(\"\u30B0\u30E9\
    \u30D5\u304C\u6709\u5411\u30B0\u30E9\u30D5\u3067\u306F\u6709\u308A\u307E\u305B\
    \u3093 (\u2565\uFE4F\u2565)\")\n\n        in_deg = self.in_deg[:]\n\n        S:\
    \ deque[int] = deque([])\n        order: List[int] = []\n\n        for i in range(self.N):\n\
    \            if in_deg[i] == 0:\n                S.append(i)\n\n        while\
    \ S:\n            if unique and len(S) != 1:\n                return [-1]\n\n\
    \            cur = S.pop()\n            order.append(cur)\n\n            for nxt\
    \ in self.get(cur):\n                in_deg[nxt] -= 1\n\n                if in_deg[nxt]\
    \ == 0:\n                    S.append(nxt)\n\n        if sum(in_deg) > 0:\n  \
    \          return [-1]\n        else:\n            return [x for x in order]\n\
    \n\nclass GraphW:\n    \"\"\"\n    \u91CD\u307F\u4ED8\u304D\u30B0\u30E9\u30D5\n\
    \    \"\"\"\n\n    def __init__(self, N: int, dire: bool = False) -> None:\n \
    \       self.N = N\n        self.dire = dire\n        self.grath = [[] for _ in\
    \ [0] * self.N]\n\n    def new_side(self, a: int, b: int, w: int):\n        \"\
    \"\"\n        \u6CE8\u610F\u30000-indexed\u304C\u524D\u63D0\n        a\u3068b\u3092\
    \u8FBA\u3067\u7E4B\u304E\u307E\u3059\n        \u6709\u5411\u30B0\u30E9\u30D5\u306A\
    \u3089\u3001a\u304B\u3089b\u3060\u3051\u3001\u7121\u5411\u30B0\u30E9\u30D5\u306A\
    \u3089\u3001a\u304B\u3089b\u3068\u3001b\u304B\u3089a\u3092\u7E4B\u304E\u307E\u3059\
    \n        \"\"\"\n        self.grath[a].append((b, w))\n        if not self.dire:\n\
    \            self.grath[b].append((a, w))\n\n    def side_input(self):\n     \
    \   \"\"\"\n        \u6A19\u6E96\u5165\u529B\u3067\u3001\u65B0\u3057\u3044\u8FBA\
    \u3092\u8FFD\u52A0\u3057\u307E\u3059\n        \"\"\"\n        a, b, w = map(lambda\
    \ x: int(x) - 1, input().split())\n        self.new_side(a, b, w + 1)\n\n    def\
    \ input(self, M: int):\n        \"\"\"\n        \u6A19\u6E96\u5165\u529B\u3067\
    \u8907\u6570\u884C\u53D7\u3051\u53D6\u308A\u3001\u5404\u884C\u306E\u5185\u5BB9\
    \u3067\u8FBA\u3092\u7E4B\u304E\u307E\u3059\n        \"\"\"\n        for _ in [0]\
    \ * M:\n            self.side_input()\n\n    def get(self, a: int) -> List[Tuple[int]]:\n\
    \        \"\"\"\n        \u9802\u70B9a\u306E\u96A3\u63A5\u9802\u70B9\u3092\u51FA\
    \u529B\u3057\u307E\u3059\n        \"\"\"\n        return self.grath[a]\n\n   \
    \ def all(self) -> List[List[Tuple[int]]]:\n        \"\"\"\n        \u30B0\u30E9\
    \u30D5\u306E\u96A3\u63A5\u30EA\u30B9\u30C8\u3092\u3059\u3079\u3066\u51FA\u529B\
    \u3057\u307E\u3059\n        \"\"\"\n        return self.grath\n\n\nfrom typing\
    \ import List\nfrom collections import defaultdict\n\n\n# UnionFind\u6728\nclass\
    \ UnionFind:\n    \"\"\"\n    rollback\u3092\u30C7\u30D5\u30A9\u30EB\u30C8\u3067\
    \u88C5\u5099\u6E08\u307F\n    \u8A08\u7B97\u91CF\u306F\u3001\u7D4C\u8DEF\u5727\
    \u7E2E\u3092\u884C\u308F\u306A\u3044\u305F\u3081\u3001\u57FA\u672C\u7684\u306A\
    UnionFind\u306E\u52D5\u4F5C\u306F\u3001\u4E00\u56DE\u3042\u305F\u308A\u3001O(log\
    \ N)\n    rollback\u306F\u3001\u4E00\u56DE\u3042\u305F\u308A\u3001O(1)\u3067\u884C\
    \u3048\u308B\u3002\n    \"\"\"\n\n    def __init__(self, n: int) -> None:\n  \
    \      self.size = n\n        self.data = [-1] * n\n        self.hist = []\n\n\
    \    def root(self, vtx: int) -> int:\n        \"\"\"\n        \u9802\u70B9vtx\u306E\
    \u89AA\u3092\u51FA\u529B\u3057\u307E\u3059\n        \"\"\"\n        if self.data[vtx]\
    \ < 0:\n            return vtx\n\n        return self.root(self.data[vtx])\n\n\
    \    def same(self, a: int, b: int):\n        \"\"\"\n        a\u3068b\u304C\u9023\
    \u7D50\u3057\u3066\u3044\u308B\u304B\u3069\u3046\u304B\u5224\u5B9A\u3057\u307E\
    \u3059\n        \"\"\"\n        return self.root(a) == self.root(b)\n\n    def\
    \ unite(self, a: int, b: int) -> bool:\n        \"\"\"\n        a\u3068b\u3092\
    \u7D50\u5408\u3057\u307E\u3059\n        root\u304C\u540C\u3058\u3067\u3082\u3001\
    \u5C65\u6B74\u306B\u306F\u8FFD\u52A0\u3057\u307E\u3059\n        \"\"\"\n     \
    \   ra, rb = self.root(a), self.root(b)\n\n        # \u5C65\u6B74\u3092\u4F5C\u6210\
    \u3059\u308B\n        new_hist = [ra, rb, self.data[ra], self.data[rb]]\n    \
    \    self.hist.append(new_hist)\n\n        if ra == rb:\n            return False\n\
    \n        if self.data[ra] > self.data[rb]:\n            ra, rb = rb, ra\n\n \
    \       self.data[ra] += self.data[rb]\n        self.data[rb] = ra\n\n       \
    \ return True\n\n    def rollback(self):\n        \"\"\"\n        undo\u3057\u307E\
    \u3059\n        redo\u306F\u3042\u308A\u307E\u305B\u3093\n        \"\"\"\n   \
    \     if not self.hist:\n            return False\n\n        ra, rb, da, db =\
    \ self.hist.pop()\n        self.data[ra] = da\n        self.data[rb] = db\n  \
    \      return True\n\n    def all(self) -> List[List[int]]:\n        D = defaultdict(list)\n\
    \n        for i in range(self.size):\n            D[self.root(i)].append(i)\n\n\
    \        res = []\n\n        for l in D.values():\n            res.append(l)\n\
    \n        return res\n\n\n# Trie\u6728\nclass Trie:\n    class Data:\n       \
    \ def __init__(self, value, ind):\n            self.count = 1\n            self.value\
    \ = value\n            self.childs = {}\n            self.ind = ind\n\n    def\
    \ __init__(self):\n        self.data = [self.Data(\"ab\", 0)]  # \u521D\u671F\u5024\
    \u306Fab\u306B\u3057\u3066\u88AB\u3089\u306A\u3044\u3088\u3046\u306B\u3059\u308B\
    \n\n    def add(self, value: str) -> int:\n        cur = 0\n        result = 0\n\
    \n        # \u518D\u5E30\u7684\u306B\u63A2\u7D22\u3059\u308B\n        for t in\
    \ value:\n            childs = self.data[cur].childs  # \u53C2\u7167\u6E21\u3057\
    \u3067\n\n            if t in childs:\n                self.data[childs[t]].count\
    \ += 1\n            else:\n                nd = self.Data(t, len(self.data))\n\
    \                childs[t] = len(self.data)\n                self.data.append(nd)\n\
    \n            result += self.data[childs[t]].count - 1\n            cur = childs[t]\n\
    \n        return result\n\n    def lcp_max(self, value: str) -> int:\n       \
    \ cur = 0\n        result = 0\n\n        for t in value:\n            childs =\
    \ self.data[cur].childs\n\n            if t not in childs:\n                break\n\
    \n            if self.data[childs[t]].count == 1:\n                break\n\n \
    \           cur = childs[t]\n            result += 1\n\n        return result\n\
    \n    def lcp_sum(self, value: str) -> int:\n        cur = 0\n        result =\
    \ 0\n\n        for t in value:\n            childs = self.data[cur].childs\n\n\
    \            if t not in childs:\n                break\n\n            if self.data[childs[t]].count\
    \ == 1:\n                break\n\n            cur = childs[t]\n            result\
    \ += self.data[childs[t]].count - 1\n\n        return result\n\n\nfrom typing\
    \ import List\n\n\nclass BIT:\n    \"\"\"\n    BIT\u3067\u3059\n    \u8981\u7D20\
    \u66F4\u65B0\u3068\u3001\u533A\u9593\u548C\u3092\u6C42\u3081\u308B\u4E8B\u304C\
    \u3067\u304D\u307E\u3059\n    1-indexed\u3067\u3059\n\n    \u8A08\u7B97\u91CF\u306F\
    \u3001\u4E00\u56DE\u306E\u52D5\u4F5C\u306B\u3064\u304D\u3059\u3079\u3066O(log\
    \ n)\u3067\u3059\n    \"\"\"\n\n    def __init__(self, n: int) -> None:\n    \
    \    self.n: int = n\n        self.bit: List[int] = [0] * (n + 1)\n\n    def sum(self,\
    \ i: int) -> int:\n        \"\"\"\n        i\u756A\u76EE\u307E\u3067\u306E\u548C\
    \u3092\u6C42\u3081\u307E\u3059\n        \u8A08\u7B97\u91CF\u306F\u3001O(log n)\u3067\
    \u3059\n        \"\"\"\n        res = 0\n\n        while i:\n            res +=\
    \ self.bit[i]\n            i -= -i & i\n\n        return res\n\n    def interval_sum(self,\
    \ l: int, r: int) -> int:\n        \"\"\"\n        l\u304B\u3089r\u307E\u3067\u306E\
    \u7DCF\u548C\u3092\u6C42\u3081\u3089\u308C\u307E\u3059\n        l\u306F0-indexed\u3067\
    \u3001r\u306F1-indexed\u306B\u3057\u3066\u304F\u3060\u3055\u3044\n        \"\"\
    \"\n        return self.sum(r) - self.sum(l)\n\n    def add(self, i: int, x: int):\n\
    \        \"\"\"\n        i\u756A\u76EE\u306E\u8981\u7D20\u306Bx\u3092\u8DB3\u3057\
    \u307E\u3059\n        \u8A08\u7B97\u91CF\u306F\u3001O(log n)\u3067\u3059\n   \
    \     \"\"\"\n        if i == 0:\n            raise IndexError(\"\u3053\u306E\u30C7\
    \u30FC\u30BF\u69CB\u9020\u306F\u30011-indexed\u3067\u3059\")\n\n        while\
    \ i <= self.n:\n            self.bit[i] += x\n            i += -i & i\n\n\nfrom\
    \ typing import Tuple\n\n\ndef euclid_dis(x1: int, y1: int, x2: int, y2: int)\
    \ -> int:\n    \"\"\"\n    \u30E6\u30FC\u30AF\u30EA\u30C3\u30C9\u8DDD\u96E2\u3092\
    \u8A08\u7B97\u3057\u307E\u3059\n\n    \u6CE8\u610F:\n    \u3053\u306E\u95A2\u6570\
    \u306Fsqrt\u3092\u53D6\u308A\u307E\u305B\u3093(\u4E3B\u306B\u5C11\u6570\u8AA4\u5DEE\
    \u7528)\n    sqrt\u3092\u53D6\u308A\u305F\u3044\u5834\u5408\u306F\u3001\u81EA\u5206\
    \u3067\u8A08\u7B97\u3057\u3066\u304F\u3060\u3055\u3044\n    \"\"\"\n\n    return\
    \ ((x1 - x2) ** 2) + ((y1 - y2) ** 2)\n\n\ndef manhattan_dis(x1: int, y1: int,\
    \ x2: int, y2: int) -> int:\n    \"\"\"\n    \u30DE\u30F3\u30CF\u30C3\u30BF\u30F3\
    \u8DDD\u96E2\u3092\u8A08\u7B97\u3057\u307E\u3059\n    \"\"\"\n\n    return abs(x1\
    \ - x2) + abs(y1 - y2)\n\n\ndef manhattan_45turn(x: int, y: int) -> Tuple[int]:\n\
    \    \"\"\"\n    \u5EA7\u6A19\u309245\u5EA6\u56DE\u8EE2\u3057\u307E\u3059\n  \
    \  \u56DE\u8EE2\u3059\u308B\u3068\u3001\u30DE\u30F3\u30CF\u30C3\u30BF\u30F3\u8DDD\
    \u96E2\u304C\u3001\u30C1\u30A7\u30D3\u30B7\u30A7\u30D5\u8DDD\u96E2\u306B\u306A\
    \u308B\u306E\u3067\u3001\u8DDD\u96E2\u306E\u6700\u5927\u5024\u306A\u3069\u304C\
    \u7C21\u5358\u306B\u6C42\u3081\u3089\u308C\u307E\u3059\n    \"\"\"\n\n    res_x\
    \ = x - y\n    res_y = x + y\n\n    return res_x, res_y\n\n\ndef chebyshev_dis(x1:\
    \ int, y1: int, x2: int, y2: int) -> int:\n    \"\"\"\n    \u30C1\u30A7\u30D3\u30B7\
    \u30A7\u30D5\u8DDD\u96E2\u3092\u8A08\u7B97\u3057\u307E\u3059\n    \"\"\"\n\n \
    \   return max(abs(x1 - x2), abs(y1 - y2))\n\n\n# \u4FBF\u5229\u5909\u6570\nINF\
    \ = 1 << 63\nlowerlist = list(\"abcdefghijklmnopqrstuvwxyz\")\nupperlist = list(\"\
    ABCDEFGHIJKLMNOPQRSTUVWXYZ\")\n\n# \u30B3\u30FC\u30C9\nN, M = il()\nL = li(M,\
    \ il, -1)\nD = defaultdict(list)\n\nUF = UnionFind(N)\n\nfor i in range(M):\n\
    \    u, v = L[i]\n    D[u].append((i, v))\n\n    UF.unite(u, v)\n\nB = UF.all()\n\
    A = []\nUF = UnionFind(N)\n\nfor l in B:\n    t = []\n    for u in l:\n      \
    \  for ind, v in D[u]:\n            if UF.same(u, v):\n                t.append((ind\
    \ + 1, u + 1))\n            else:\n                UF.unite(u, v)\n\n    A.append((l[0]\
    \ + 1, len(t), t))\n\nA.sort(key=lambda x: x[1])\nQ = deque()\nQ.append(A.pop())\n\
    \nans = []\n\nwhile Q and A:\n    ou, l, t = Q.popleft()\n    ind, u = t.pop()\n\
    \    l -= 1\n\n    v, nl, nt = A.pop()\n    ans.append((ind, u, v))\n\n    if\
    \ nl > 0:\n        Q.append((v, nl, nt))\n\n    if l > 0:\n        Q.append((ou,\
    \ l, t))\n\nprint(len(ans))\n\nfor l in ans:\n    print(*l)\n"
  dependsOn: []
  isVerificationFile: false
  path: tests/ABC392E.py
  requiredBy: []
  timestamp: '1970-01-01 00:00:00+00:00'
  verificationStatus: LIBRARY_NO_TESTS
  verifiedWith: []
documentation_of: tests/ABC392E.py
layout: document
redirect_from:
- /library/tests/ABC392E.py
- /library/tests/ABC392E.py.html
title: tests/ABC392E.py
---
