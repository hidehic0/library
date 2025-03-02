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
    \u30D6\u30E9\u30EA\nfrom collections import deque, defaultdict, Counter\nfrom\
    \ math import pi, gcd, lcm\nfrom itertools import permutations\nimport bisect\n\
    import sys\nimport heapq\nfrom typing import List, Any\nimport unittest\n\n# from\
    \ atcoder.segtree import SegTree\n# from atcoder.lazysegtree import LazySegTree\n\
    # from atcoder.dsu import DSU\n\n# cortedcontainers\u306F\u4F7F\u3046\u3068\u304D\
    \u3060\u3051 wandbox\u975E\u5BFE\u5FDC\u306A\u306E\u3067\n# from sortedcontainers\
    \ import SortedDict, SortedSet, SortedList\n\n# import pypyjit\n# pypyjit.set_param(\"\
    max_unroll_recursion=-1\")\n\nsys.setrecursionlimit(5 * 10**5)\n\n\n# \u95A2\u6570\
    \ndef is_prime(n):\n    if n == 1:\n        return False\n\n    def f(a, t, n):\n\
    \        x = pow(a, t, n)\n        nt = n - 1\n        while t != nt and x !=\
    \ 1 and x != nt:\n            x = pow(x, 2, n)\n            t <<= 1\n\n      \
    \  return t & 1 or x == nt\n\n    if n == 2:\n        return True\n    elif n\
    \ % 2 == 0:\n        return False\n\n    d = n - 1\n    d >>= 1\n\n    while d\
    \ & 1 == 0:\n        d >>= 1\n\n    checklist = (\n        [2, 7, 61] if 2**32\
    \ > n else [2, 325, 9375, 28178, 450775, 9780504, 1795265022]\n    )\n\n    for\
    \ i in checklist:\n        if i >= n:\n            break\n        if not f(i,\
    \ d, n):\n            return False\n\n    return True\n\n\ndef eratosthenes(n):\n\
    \    primes = [True] * (n + 1)\n    primes[0], primes[1] = False, False\n    i\
    \ = 2\n    while i**2 <= n:\n        if primes[i]:\n            for k in range(i\
    \ * 2, n + 1, i):\n                primes[k] = False\n\n        i += 1\n\n   \
    \ return [i for i, p in enumerate(primes) if p]\n\n\ndef calc_divisors(N):\n \
    \   # \u7D04\u6570\u5168\u5217\u6319\n    result = []\n\n    for i in range(1,\
    \ N + 1):\n        if i * i > N:\n            break\n\n        if N % i != 0:\n\
    \            continue\n\n        heapq.heappush(result, i)\n        if N // i\
    \ != i:\n            heapq.heappush(result, N // i)\n\n    return result\n\n\n\
    def factorization(n):\n    # \u7D20\u56E0\u6570\u5206\u89E3\n    result = []\n\
    \    tmp = n\n    for i in range(2, int(-(-(n**0.5) // 1)) + 1):\n        if tmp\
    \ % i == 0:\n            cnt = 0\n            while tmp % i == 0:\n          \
    \      cnt += 1\n                tmp //= i\n            result.append([i, cnt])\n\
    \n    if tmp != 1:\n        result.append([tmp, 1])\n\n    if result == []:\n\
    \        result.append([n, 1])\n\n    return result\n\n\nclass TestMathFunctions(unittest.TestCase):\n\
    \    def test_is_prime(self):\n        test_cases = [\n            (1, False),\n\
    \            (2, True),\n            (3, True),\n            (4, False),\n   \
    \         (5, True),\n            (6, False),\n            (1747, True),\n   \
    \         (256, False),\n        ]\n\n        for i, ans in test_cases:\n    \
    \        with self.subTest(i=i):\n                self.assertEqual(is_prime(i),\
    \ ans)\n\n\ndef create_array2(a: int, b: int, default: Any = 0) -> List[List[Any]]:\n\
    \    \"\"\"\n    \uFF12\u6B21\u5143\u914D\u5217\u3092\u521D\u671F\u5316\u3059\u308B\
    \u95A2\u6570\n    \"\"\"\n    return [[default] * b for _ in [0] * a]\n\n\ndef\
    \ create_array3(a: int, b: int, c: int, default: Any = 0) -> List[List[List[Any]]]:\n\
    \    \"\"\"\n    \uFF13\u6B21\u5143\u914D\u5217\u3092\u521D\u671F\u5316\u3059\u308B\
    \u95A2\u6570\n    \"\"\"\n    return [[[default] * c for _ in [0] * b] for _ in\
    \ [0] * a]\n\n\n# \u6A19\u6E96\u5165\u529B\u7CFB\n# \u4E00\u884C\u306B\u4E00\u3064\
    \u306Estring\ndef s():\n    return sys.stdin.readline().rstrip()\n\n\n# \u4E00\
    \u884C\u306B\u8907\u6570\u306Estring\ndef sl():\n    return s().split()\n\n\n\
    # \u4E00\u3064\u306Eint\ndef ii():\n    return int(s())\n\n\n# \u4E00\u884C\u306B\
    \u8907\u6570\u306Eint\ndef il(add_num: int = 0):\n    return list(map(lambda i:\
    \ int(i) + add_num, sl()))\n\n\n# \u8907\u6570\u884C\u306E\u5165\u529B\u3092\u30B5\
    \u30DD\u30FC\u30C8\ndef li(n: int, func, *args):\n    return [func(*args) for\
    \ _ in [0] * n]\n\n\n# ac-library\u7528\u30E1\u30E2\n\"\"\"\nsegtree\n\n\u521D\
    \u671F\u5316\u3059\u308B\u3068\u304D\nSegtree(op,e,v)\n\nop\u306F\u30DE\u30FC\u30B8\
    \u3059\u308B\u95A2\u6570\n\u4F8B\n\ndef op(a,b):\n    return a+b\n\ne\u306F\u521D\
    \u671F\u5316\u3059\u308B\u5024\n\nv\u306F\u914D\u5217\u306E\u9577\u3055\u307E\u305F\
    \u306F\u3001\u521D\u671F\u5316\u3059\u308B\u5185\u5BB9\n\"\"\"\n\n\n# \u7121\u5411\
    \u30B0\u30E9\u30D5\nclass Graph:\n    def __init__(self, N: int, dire: bool =\
    \ False) -> None:\n        self.N = N\n        self.dire = dire\n        self.grath\
    \ = [[] for _ in [0] * self.N]\n        self.in_deg = [0] * N\n\n    def new_side(self,\
    \ a: int, b: int):\n        # \u6CE8\u610F\u30000-indexed\u304C\u524D\u63D0\n\
    \        self.grath[a].append(b)\n        if self.dire:\n            self.in_deg[b]\
    \ += 1\n\n        if not self.dire:\n            self.grath[b].append(a)\n\n \
    \   def side_input(self):\n        # \u65B0\u3057\u3044\u8FBA\u3092input\n   \
    \     a, b = il(-1)\n        self.new_side(a, b)\n\n    def input(self, M: int):\n\
    \        # \u8907\u6570\u884C\u306E\u8FBA\u306Einput\n        for _ in [0] * M:\n\
    \            self.side_input()\n\n    def get(self, a: int):\n        # \u9802\
    \u70B9a\u306E\u96A3\u63A5\u70B9\u3092\u51FA\u529B\n        return self.grath[a]\n\
    \n    def all(self):\n        # \u30B0\u30E9\u30D5\u306E\u5185\u5BB9\u3092\u3059\
    \u3079\u3066\u51FA\u529B\n        return self.grath\n\n    def topological(self,\
    \ unique: bool = False):\n        if not self.dire:\n            raise ValueError(\"\
    \u30B0\u30E9\u30D5\u304C\u6709\u5411\u30B0\u30E9\u30D5\u3067\u306F\u6709\u308A\
    \u307E\u305B\u3093 (\u2565\uFE4F\u2565)\")\n\n        in_deg = self.in_deg[:]\n\
    \n        S: deque[int] = deque([])\n        order: List[int] = []\n\n       \
    \ for i in range(self.N):\n            if in_deg[i] == 0:\n                S.append(i)\n\
    \n        while S:\n            if unique and len(S) != 1:\n                return\
    \ [-1]\n\n            cur = S.pop()\n            order.append(cur)\n\n       \
    \     for nxt in self.get(cur):\n                in_deg[nxt] -= 1\n\n        \
    \        if in_deg[nxt] == 0:\n                    S.append(nxt)\n\n        if\
    \ sum(in_deg) > 0:\n            return [-1]\n        else:\n            return\
    \ [x for x in order]\n\n\n# \u91CD\u307F\u4ED8\u304D\u30B0\u30E9\u30D5\nclass\
    \ GraphW:\n    def __init__(self, N: int, dire: bool = False) -> None:\n     \
    \   self.N = N\n        self.dire = dire\n        self.grath = [[] for _ in [0]\
    \ * self.N]\n\n    def new_side(self, a: int, b: int, w: int):\n        # \u6CE8\
    \u610F\u30000-indexed\u304C\u524D\u63D0\n        self.grath[a].append((b, w))\n\
    \        if not self.dire:\n            self.grath[b].append((a, w))\n\n    def\
    \ side_input(self):\n        # \u65B0\u3057\u3044\u8FBA\u3092input\n        a,\
    \ b, w = il(-1)\n        self.new_side(a, b, w)\n\n    def input(self, M: int):\n\
    \        # \u8907\u6570\u884C\u306E\u8FBA\u306Einput\n        for _ in [0] * M:\n\
    \            self.side_input()\n\n    def get(self, a: int):\n        # \u9802\
    \u70B9a\u306E\u96A3\u63A5\u70B9\u3092\u51FA\u529B\n        return self.grath[a]\n\
    \n    def all(self):\n        # \u30B0\u30E9\u30D5\u306E\u5185\u5BB9\u3092\u3059\
    \u3079\u3066\u51FA\u529B\n        return self.grath\n\n\nclass Trie:\n    class\
    \ Data:\n        def __init__(self, value, ind):\n            self.count = 1\n\
    \            self.value = value\n            self.childs = {}\n            self.ind\
    \ = ind\n\n    def __init__(self):\n        self.data = [self.Data(\"ab\", 0)]\
    \  # \u521D\u671F\u5024\u306Fab\u306B\u3057\u3066\u88AB\u3089\u306A\u3044\u3088\
    \u3046\u306B\u3059\u308B\n\n    def add(self, value: str) -> None:\n        cur\
    \ = 0\n\n        # \u518D\u5E30\u7684\u306B\u63A2\u7D22\u3059\u308B\n        for\
    \ t in value:\n            childs = self.data[cur].childs  # \u53C2\u7167\u6E21\
    \u3057\u3067\n\n            if t in childs:\n                self.data[childs[t]].count\
    \ += 1\n            else:\n                nd = self.Data(t, len(self.data))\n\
    \                childs[t] = len(self.data)\n                self.data.append(nd)\n\
    \n            cur = childs[t]\n\n        return None\n\n    def lcp_max(self,\
    \ value: str) -> int:\n        cur = 0\n        result = 0\n\n        for t in\
    \ value:\n            childs = self.data[cur].childs\n\n            if t not in\
    \ childs:\n                break\n\n            if self.data[childs[t]].count\
    \ == 1:\n                break\n\n            cur = childs[t]\n            result\
    \ += 1\n\n        return result\n\n    def lcp_sum(self, value: str) -> int:\n\
    \        cur = 0\n        result = 0\n\n        for t in value:\n            childs\
    \ = self.data[cur].childs\n\n            if t not in childs:\n               \
    \ break\n\n            if self.data[childs[t]].count == 1:\n                break\n\
    \n            cur = childs[t]\n            result += self.data[childs[t]].count\
    \ - 1\n\n        return result\n\n\n# \u4FBF\u5229\u5909\u6570\nINF = 1 << 63\n\
    lowerlist = list(\"abcdefghijklmnopqrstuvwxyz\")\nupperlist = list(\"ABCDEFGHIJKLMNOPQRSTUVWXYZ\"\
    )\n\n# \u30C6\u30B9\u30C8\u3092\u5B9F\u884C\u3059\u308B\nif sys.argv == [\"code/main.py\"\
    ]:\n    unittest.main()\n\n# \u30B3\u30FC\u30C9\nN = ii()\nL = []\nTR = Trie()\n\
    \nfor _ in [0] * N:\n    S = s()\n    L.append(S)\n    TR.add(S)\n\nfor S in L:\n\
    \    print(TR.lcp_max(S))\n"
  dependsOn: []
  isVerificationFile: false
  path: tests/ABC287E.py
  requiredBy: []
  timestamp: '1970-01-01 00:00:00+00:00'
  verificationStatus: LIBRARY_NO_TESTS
  verifiedWith: []
documentation_of: tests/ABC287E.py
layout: document
redirect_from:
- /library/tests/ABC287E.py
- /library/tests/ABC287E.py.html
title: tests/ABC287E.py
---
