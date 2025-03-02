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
  code: "from typing import Callable\n\n\ndef binary_search(\n    fn: Callable[[int],\
    \ bool], right: int = 0, left: int = -1, return_left: bool = True\n) -> int:\n\
    \    \"\"\"\n    \u4E8C\u5206\u63A2\u7D22\u306E\u62BD\u8C61\u7684\u306A\u30E9\u30A4\
    \u30D6\u30E9\u30EA\n    \u8A55\u4FA1\u95A2\u6570\u306E\u7D50\u679C\u306B\u5FDC\
    \u3058\u3066\u3001\u4E8C\u5206\u63A2\u7D22\u3059\u308B\n    \u6700\u7D42\u7684\
    \u306B\u306Fleft\u3092\u51FA\u529B\u3057\u307E\u3059\n\n    \u95A2\u6570\u306E\
    \u30C6\u30F3\u30D7\u30EC\u30FC\u30C8\n    def check(mid:int):\n        if A[mid]\
    \ > x:\n            return True\n        else:\n            return False\n\n \
    \   mid\u306F\u5FC5\u9808\u3067\u3059\u3002\u305D\u308C\u4EE5\u5916\u306F\u3054\
    \u81EA\u7531\u306B\u3069\u3046\u305E\n    \"\"\"\n    while right - left > 1:\n\
    \        mid = (left + right) // 2\n\n        if fn(mid):\n            left =\
    \ mid\n        else:\n            right = mid\n\n    return left if return_left\
    \ else right\n"
  dependsOn: []
  isVerificationFile: false
  path: libs/binary_search.py
  requiredBy: []
  timestamp: '1970-01-01 00:00:00+00:00'
  verificationStatus: LIBRARY_NO_TESTS
  verifiedWith: []
documentation_of: libs/binary_search.py
layout: document
redirect_from:
- /library/libs/binary_search.py
- /library/libs/binary_search.py.html
title: libs/binary_search.py
---
