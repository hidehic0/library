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
  code: "from typing import List\nfrom collections import defaultdict\n\n\n# UnionFind\u6728\
    \nclass UnionFind:\n    \"\"\"\n    rollback\u3092\u30C7\u30D5\u30A9\u30EB\u30C8\
    \u3067\u88C5\u5099\u6E08\u307F\n    \u8A08\u7B97\u91CF\u306F\u3001\u7D4C\u8DEF\
    \u5727\u7E2E\u3092\u884C\u308F\u306A\u3044\u305F\u3081\u3001\u57FA\u672C\u7684\
    \u306AUnionFind\u306E\u52D5\u4F5C\u306F\u3001\u4E00\u56DE\u3042\u305F\u308A\u3001\
    O(log N)\n    rollback\u306F\u3001\u4E00\u56DE\u3042\u305F\u308A\u3001O(1)\u3067\
    \u884C\u3048\u308B\u3002\n    \"\"\"\n\n    def __init__(self, n: int) -> None:\n\
    \        self.size = n\n        self.data = [-1] * n\n        self.hist = []\n\
    \n    def root(self, vtx: int) -> int:\n        \"\"\"\n        \u9802\u70B9vtx\u306E\
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
    \n        return res\n"
  dependsOn: []
  isVerificationFile: false
  path: libs/unionfind.py
  requiredBy: []
  timestamp: '1970-01-01 00:00:00+00:00'
  verificationStatus: LIBRARY_NO_TESTS
  verifiedWith: []
documentation_of: libs/unionfind.py
layout: document
redirect_from:
- /library/libs/unionfind.py
- /library/libs/unionfind.py.html
title: libs/unionfind.py
---
