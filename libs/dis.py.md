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
  code: "from typing import Tuple\n\n\ndef euclid_dis(x1: int, y1: int, x2: int, y2:\
    \ int) -> int:\n    \"\"\"\n    \u30E6\u30FC\u30AF\u30EA\u30C3\u30C9\u8DDD\u96E2\
    \u3092\u8A08\u7B97\u3057\u307E\u3059\n\n    \u6CE8\u610F:\n    \u3053\u306E\u95A2\
    \u6570\u306Fsqrt\u3092\u53D6\u308A\u307E\u305B\u3093(\u4E3B\u306B\u5C11\u6570\u8AA4\
    \u5DEE\u7528)\n    sqrt\u3092\u53D6\u308A\u305F\u3044\u5834\u5408\u306F\u3001\u81EA\
    \u5206\u3067\u8A08\u7B97\u3057\u3066\u304F\u3060\u3055\u3044\n    \"\"\"\n\n \
    \   return ((x1 - x2) ** 2) + ((y1 - y2) ** 2)\n\n\ndef manhattan_dis(x1: int,\
    \ y1: int, x2: int, y2: int) -> int:\n    \"\"\"\n    \u30DE\u30F3\u30CF\u30C3\
    \u30BF\u30F3\u8DDD\u96E2\u3092\u8A08\u7B97\u3057\u307E\u3059\n    \"\"\"\n\n \
    \   return abs(x1 - x2) + abs(y1 - y2)\n\n\ndef manhattan_45turn(x: int, y: int)\
    \ -> Tuple[int]:\n    \"\"\"\n    \u5EA7\u6A19\u309245\u5EA6\u56DE\u8EE2\u3057\
    \u307E\u3059\n    \u56DE\u8EE2\u3059\u308B\u3068\u3001\u30DE\u30F3\u30CF\u30C3\
    \u30BF\u30F3\u8DDD\u96E2\u304C\u3001\u30C1\u30A7\u30D3\u30B7\u30A7\u30D5\u8DDD\
    \u96E2\u306B\u306A\u308B\u306E\u3067\u3001\u8DDD\u96E2\u306E\u6700\u5927\u5024\
    \u306A\u3069\u304C\u7C21\u5358\u306B\u6C42\u3081\u3089\u308C\u307E\u3059\n   \
    \ \"\"\"\n\n    res_x = x - y\n    res_y = x + y\n\n    return res_x, res_y\n\n\
    \ndef chebyshev_dis(x1: int, y1: int, x2: int, y2: int) -> int:\n    \"\"\"\n\
    \    \u30C1\u30A7\u30D3\u30B7\u30A7\u30D5\u8DDD\u96E2\u3092\u8A08\u7B97\u3057\u307E\
    \u3059\n    \"\"\"\n\n    return max(abs(x1 - x2), abs(y1 - y2))\n"
  dependsOn: []
  isVerificationFile: false
  path: libs/dis.py
  requiredBy: []
  timestamp: '1970-01-01 00:00:00+00:00'
  verificationStatus: LIBRARY_NO_TESTS
  verifiedWith: []
documentation_of: libs/dis.py
layout: document
redirect_from:
- /library/libs/dis.py
- /library/libs/dis.py.html
title: libs/dis.py
---
