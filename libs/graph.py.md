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
  code: "# \u30B0\u30E9\u30D5\u69CB\u9020\n# \u7121\u5411\u30B0\u30E9\u30D5\nfrom\
    \ collections import deque\nfrom typing import List, Tuple\n\n\nclass Graph:\n\
    \    \"\"\"\n    \u30B0\u30E9\u30D5\u69CB\u9020\u4F53\n    \"\"\"\n\n    def __init__(self,\
    \ N: int, dire: bool = False) -> None:\n        \"\"\"\n        N\u306F\u9802\u70B9\
    \u6570\u3001dire\u306F\u6709\u5411\u30B0\u30E9\u30D5\u304B\u3067\u3059\n     \
    \   \"\"\"\n        self.N = N\n        self.dire = dire\n        self.grath =\
    \ [[] for _ in [0] * self.N]\n        self.in_deg = [0] * N\n\n    def new_side(self,\
    \ a: int, b: int):\n        \"\"\"\n        \u6CE8\u610F\u30000-indexed\u304C\u524D\
    \u63D0\n        a\u3068b\u3092\u8FBA\u3067\u7E4B\u304E\u307E\u3059\n        \u6709\
    \u5411\u30B0\u30E9\u30D5\u306A\u3089\u3001a\u304B\u3089b\u3060\u3051\u3001\u7121\
    \u5411\u30B0\u30E9\u30D5\u306A\u3089\u3001a\u304B\u3089b\u3068\u3001b\u304B\u3089\
    a\u3092\u7E4B\u304E\u307E\u3059\n        \"\"\"\n        self.grath[a].append(b)\n\
    \        if self.dire:\n            self.in_deg[b] += 1\n\n        if not self.dire:\n\
    \            self.grath[b].append(a)\n\n    def side_input(self):\n        \"\"\
    \"\n        \u6A19\u6E96\u5165\u529B\u3067\u3001\u65B0\u3057\u3044\u8FBA\u3092\
    \u8FFD\u52A0\u3057\u307E\u3059\n        \"\"\"\n        a, b = map(lambda x: int(x)\
    \ - 1, input().split())\n        self.new_side(a, b)\n\n    def input(self, M:\
    \ int):\n        \"\"\"\n        \u6A19\u6E96\u5165\u529B\u3067\u8907\u6570\u884C\
    \u53D7\u3051\u53D6\u308A\u3001\u5404\u884C\u306E\u5185\u5BB9\u3067\u8FBA\u3092\
    \u7E4B\u304E\u307E\u3059\n        \"\"\"\n        for _ in [0] * M:\n        \
    \    self.side_input()\n\n    def get(self, a: int):\n        \"\"\"\n       \
    \ \u9802\u70B9a\u306E\u96A3\u63A5\u9802\u70B9\u3092\u51FA\u529B\u3057\u307E\u3059\
    \n        \"\"\"\n        return self.grath[a]\n\n    def all(self) -> List[List[int]]:\n\
    \        \"\"\"\n        \u30B0\u30E9\u30D5\u306E\u96A3\u63A5\u30EA\u30B9\u30C8\
    \u3092\u3059\u3079\u3066\u51FA\u529B\u3057\u307E\u3059\n        \"\"\"\n     \
    \   return self.grath\n\n    def topological(self, unique: bool = False) -> List[int]:\n\
    \        \"\"\"\n        \u30C8\u30DD\u30ED\u30B8\u30AB\u30EB\u30BD\u30FC\u30C8\
    \u3057\u307E\u3059\n        \u6709\u5411\u30B0\u30E9\u30D5\u9650\u5B9A\u3067\u3059\
    \n\n        \u5F15\u6570\u306Eunique\u306F\u3001\u30C8\u30DD\u30ED\u30B8\u30AB\
    \u30EB\u30BD\u30FC\u30C8\u7D50\u679C\u304C\u3001\u4E00\u610F\u306B\u5B9A\u307E\
    \u3089\u306A\u3044\u3068\u30A8\u30E9\u30FC\u3092\u5410\u304D\u307E\u3059\n   \
    \     \u9589\u8DEF\u304C\u3042\u308B\u3001\u307E\u305F\u306F\u3001unique\u304C\
    True\u3067\u4E00\u610F\u306B\u5B9A\u307E\u3089\u306A\u304B\u3063\u305F\u6642\u306F\
    \u3001[-1]\u3092\u8FD4\u3057\u307E\u3059\n        \"\"\"\n        if not self.dire:\n\
    \            raise ValueError(\"\u30B0\u30E9\u30D5\u304C\u6709\u5411\u30B0\u30E9\
    \u30D5\u3067\u306F\u6709\u308A\u307E\u305B\u3093 (\u2565\uFE4F\u2565)\")\n\n \
    \       in_deg = self.in_deg[:]\n\n        S: deque[int] = deque([])\n       \
    \ order: List[int] = []\n\n        for i in range(self.N):\n            if in_deg[i]\
    \ == 0:\n                S.append(i)\n\n        while S:\n            if unique\
    \ and len(S) != 1:\n                return [-1]\n\n            cur = S.pop()\n\
    \            order.append(cur)\n\n            for nxt in self.get(cur):\n    \
    \            in_deg[nxt] -= 1\n\n                if in_deg[nxt] == 0:\n      \
    \              S.append(nxt)\n\n        if sum(in_deg) > 0:\n            return\
    \ [-1]\n        else:\n            return [x for x in order]\n\n\nclass GraphW:\n\
    \    \"\"\"\n    \u91CD\u307F\u4ED8\u304D\u30B0\u30E9\u30D5\n    \"\"\"\n\n  \
    \  def __init__(self, N: int, dire: bool = False) -> None:\n        self.N = N\n\
    \        self.dire = dire\n        self.grath = [[] for _ in [0] * self.N]\n\n\
    \    def new_side(self, a: int, b: int, w: int):\n        \"\"\"\n        \u6CE8\
    \u610F\u30000-indexed\u304C\u524D\u63D0\n        a\u3068b\u3092\u8FBA\u3067\u7E4B\
    \u304E\u307E\u3059\n        \u6709\u5411\u30B0\u30E9\u30D5\u306A\u3089\u3001a\u304B\
    \u3089b\u3060\u3051\u3001\u7121\u5411\u30B0\u30E9\u30D5\u306A\u3089\u3001a\u304B\
    \u3089b\u3068\u3001b\u304B\u3089a\u3092\u7E4B\u304E\u307E\u3059\n        \"\"\"\
    \n        self.grath[a].append((b, w))\n        if not self.dire:\n          \
    \  self.grath[b].append((a, w))\n\n    def side_input(self):\n        \"\"\"\n\
    \        \u6A19\u6E96\u5165\u529B\u3067\u3001\u65B0\u3057\u3044\u8FBA\u3092\u8FFD\
    \u52A0\u3057\u307E\u3059\n        \"\"\"\n        a, b, w = map(lambda x: int(x)\
    \ - 1, input().split())\n        self.new_side(a, b, w + 1)\n\n    def input(self,\
    \ M: int):\n        \"\"\"\n        \u6A19\u6E96\u5165\u529B\u3067\u8907\u6570\
    \u884C\u53D7\u3051\u53D6\u308A\u3001\u5404\u884C\u306E\u5185\u5BB9\u3067\u8FBA\
    \u3092\u7E4B\u304E\u307E\u3059\n        \"\"\"\n        for _ in [0] * M:\n  \
    \          self.side_input()\n\n    def get(self, a: int) -> List[Tuple[int]]:\n\
    \        \"\"\"\n        \u9802\u70B9a\u306E\u96A3\u63A5\u9802\u70B9\u3092\u51FA\
    \u529B\u3057\u307E\u3059\n        \"\"\"\n        return self.grath[a]\n\n   \
    \ def all(self) -> List[List[Tuple[int]]]:\n        \"\"\"\n        \u30B0\u30E9\
    \u30D5\u306E\u96A3\u63A5\u30EA\u30B9\u30C8\u3092\u3059\u3079\u3066\u51FA\u529B\
    \u3057\u307E\u3059\n        \"\"\"\n        return self.grath\n"
  dependsOn: []
  isVerificationFile: false
  path: libs/graph.py
  requiredBy: []
  timestamp: '1970-01-01 00:00:00+00:00'
  verificationStatus: LIBRARY_NO_TESTS
  verifiedWith: []
documentation_of: libs/graph.py
layout: document
redirect_from:
- /library/libs/graph.py
- /library/libs/graph.py.html
title: libs/graph.py
---
