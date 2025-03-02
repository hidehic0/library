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
  code: "from typing import List\n\n\nclass BIT:\n    \"\"\"\n    BIT\u3067\u3059\n\
    \    \u8981\u7D20\u66F4\u65B0\u3068\u3001\u533A\u9593\u548C\u3092\u6C42\u3081\u308B\
    \u4E8B\u304C\u3067\u304D\u307E\u3059\n    1-indexed\u3067\u3059\n\n    \u8A08\u7B97\
    \u91CF\u306F\u3001\u4E00\u56DE\u306E\u52D5\u4F5C\u306B\u3064\u304D\u3059\u3079\
    \u3066O(log n)\u3067\u3059\n    \"\"\"\n\n    def __init__(self, n: int) -> None:\n\
    \        self.n: int = n\n        self.bit: List[int] = [0] * (n + 1)\n\n    def\
    \ sum(self, i: int) -> int:\n        \"\"\"\n        i\u756A\u76EE\u307E\u3067\
    \u306E\u548C\u3092\u6C42\u3081\u307E\u3059\n        \u8A08\u7B97\u91CF\u306F\u3001\
    O(log n)\u3067\u3059\n        \"\"\"\n\n        res = 0\n\n        while i:\n\
    \            res += self.bit[i]\n            i -= -i & i\n\n        return res\n\
    \n    def interval_sum(self, l: int, r: int) -> None:\n        \"\"\"\n      \
    \  l\u304B\u3089r\u307E\u3067\u306E\u7DCF\u548C\u3092\u6C42\u3081\u3089\u308C\u307E\
    \u3059\n        l\u306F0-indexed\u3067\u3001r\u306F1-indexed\u306B\u3057\u3066\
    \u304F\u3060\u3055\u3044\n        \"\"\"\n        return self.sum(r) - self.sum(l)\n\
    \n    def add(self, i: int, x: int):\n        \"\"\"\n        i\u756A\u76EE\u306E\
    \u8981\u7D20\u306Bx\u3092\u8DB3\u3057\u307E\u3059\n        \u8A08\u7B97\u91CF\u306F\
    \u3001O(log n)\u3067\u3059\n        \"\"\"\n        if not (0 < i <= self.n):\n\
    \            raise IndexError(\"i\u304C\u7BC4\u56F2\u306B\u53CE\u307E\u3063\u3066\
    \u3044\u307E\u305B\u3093\")\n\n        if i == 0:\n            raise IndexError(\"\
    \u3053\u306E\u30C7\u30FC\u30BF\u69CB\u9020\u306F\u30011-indexed\u3067\u3059\"\
    )\n\n        while i <= self.n:\n            self.bit[i] += x\n            i +=\
    \ -i & i\n\n\nN, Q = list(map(int, input().split()))\nbit = BIT(N)\n\nfor i, a\
    \ in enumerate(list(map(int, input().split()))):\n    bit.add(i + 1, a)\n\nfor\
    \ _ in [0] * Q:\n    l, r = list(map(int, input().split()))\n\n    print(bit.interval_sum(l\
    \ - 1, r))\n"
  dependsOn: []
  isVerificationFile: false
  path: tests/tessoku_A06.py
  requiredBy: []
  timestamp: '1970-01-01 00:00:00+00:00'
  verificationStatus: LIBRARY_NO_TESTS
  verifiedWith: []
documentation_of: tests/tessoku_A06.py
layout: document
redirect_from:
- /library/tests/tessoku_A06.py
- /library/tests/tessoku_A06.py.html
title: tests/tessoku_A06.py
---
