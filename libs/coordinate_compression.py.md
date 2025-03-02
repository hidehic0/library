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
  code: "from typing import List, Tuple\n\n\ndef coordinate_compression(lis: List[int]\
    \ | Tuple[int]) -> List[int]:\n    \"\"\"\n    \u5EA7\u6A19\u5727\u7E2E\u3057\u307E\
    \u3059\n    \u8A08\u7B97\u91CF\u306F\u3001O(N log N)\u3067\u3059\n\n    l\u3068\
    r\u306F\u3001\u307E\u3068\u3081\u3066\u5165\u308C\u308B\u4E8B\u3067\u3001\u5EA7\
    \u5727\u3067\u304D\u307E\u3059\n    \"\"\"\n    res = []\n    d = {num: ind for\
    \ ind, num in enumerate(sorted(set(lis)))}\n\n    for a in lis:\n        res.append(d[a])\n\
    \n    return res\n"
  dependsOn: []
  isVerificationFile: false
  path: libs/coordinate_compression.py
  requiredBy: []
  timestamp: '1970-01-01 00:00:00+00:00'
  verificationStatus: LIBRARY_NO_TESTS
  verifiedWith: []
documentation_of: libs/coordinate_compression.py
layout: document
redirect_from:
- /library/libs/coordinate_compression.py
- /library/libs/coordinate_compression.py.html
title: libs/coordinate_compression.py
---
