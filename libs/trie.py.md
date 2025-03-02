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
  code: "# Trie\u6728\nclass Trie:\n    class Data:\n        def __init__(self, value,\
    \ ind):\n            self.count = 1\n            self.value = value\n        \
    \    self.childs = {}\n            self.ind = ind\n\n    def __init__(self):\n\
    \        self.data = [self.Data(\"ab\", 0)]  # \u521D\u671F\u5024\u306Fab\u306B\
    \u3057\u3066\u88AB\u3089\u306A\u3044\u3088\u3046\u306B\u3059\u308B\n\n    def\
    \ add(self, value: str) -> int:\n        cur = 0\n        result = 0\n\n     \
    \   # \u518D\u5E30\u7684\u306B\u63A2\u7D22\u3059\u308B\n        for t in value:\n\
    \            childs = self.data[cur].childs  # \u53C2\u7167\u6E21\u3057\u3067\n\
    \n            if t in childs:\n                self.data[childs[t]].count += 1\n\
    \            else:\n                nd = self.Data(t, len(self.data))\n      \
    \          childs[t] = len(self.data)\n                self.data.append(nd)\n\n\
    \            result += self.data[childs[t]].count - 1\n            cur = childs[t]\n\
    \n        return result\n\n    def lcp_max(self, value: str) -> int:\n       \
    \ cur = 0\n        result = 0\n\n        for t in value:\n            childs =\
    \ self.data[cur].childs\n\n            if t not in childs:\n                break\n\
    \n            if self.data[childs[t]].count == 1:\n                break\n\n \
    \           cur = childs[t]\n            result += 1\n\n        return result\n\
    \n    def lcp_sum(self, value: str) -> int:\n        cur = 0\n        result =\
    \ 0\n\n        for t in value:\n            childs = self.data[cur].childs\n\n\
    \            if t not in childs:\n                break\n\n            if self.data[childs[t]].count\
    \ == 1:\n                break\n\n            cur = childs[t]\n            result\
    \ += self.data[childs[t]].count - 1\n\n        return result\n"
  dependsOn: []
  isVerificationFile: false
  path: libs/trie.py
  requiredBy: []
  timestamp: '2025-03-02 19:35:59+09:00'
  verificationStatus: LIBRARY_NO_TESTS
  verifiedWith: []
documentation_of: libs/trie.py
layout: document
redirect_from:
- /library/libs/trie.py
- /library/libs/trie.py.html
title: libs/trie.py
---
