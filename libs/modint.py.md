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
  code: "def mod_add(a: int, b: int, mod: int):\n    \"\"\"\n    \u8DB3\u3057\u7B97\
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
    \ -> bool:\n        return self.rhs(rhs) != self.x\n"
  dependsOn: []
  isVerificationFile: false
  path: libs/modint.py
  requiredBy: []
  timestamp: '2025-03-02 19:35:59+09:00'
  verificationStatus: LIBRARY_NO_TESTS
  verifiedWith: []
documentation_of: libs/modint.py
layout: document
redirect_from:
- /library/libs/modint.py
- /library/libs/modint.py.html
title: libs/modint.py
---
