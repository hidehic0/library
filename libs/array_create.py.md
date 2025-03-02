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
  code: "# \u591A\u6B21\u5143\u914D\u5217\u4F5C\u6210\nfrom typing import List, Any\n\
    \n\ndef create_array1(n: int, default: Any = 0) -> List[Any]:\n    \"\"\"\n  \
    \  1\u6B21\u5143\u914D\u5217\u3092\u521D\u671F\u5316\u3059\u308B\u95A2\u6570\n\
    \    \"\"\"\n    return [default] * n\n\n\ndef create_array2(a: int, b: int, default:\
    \ Any = 0) -> List[List[Any]]:\n    \"\"\"\n    2\u6B21\u5143\u914D\u5217\u3092\
    \u521D\u671F\u5316\u3059\u308B\u95A2\u6570\n    \"\"\"\n    return [[default]\
    \ * b for _ in [0] * a]\n\n\ndef create_array3(a: int, b: int, c: int, default:\
    \ Any = 0) -> List[List[List[Any]]]:\n    \"\"\"\n    3\u6B21\u5143\u914D\u5217\
    \u3092\u521D\u671F\u5316\u3059\u308B\u95A2\u6570\n    \"\"\"\n    return [[[default]\
    \ * c for _ in [0] * b] for _ in [0] * a]\n"
  dependsOn: []
  isVerificationFile: false
  path: libs/array_create.py
  requiredBy: []
  timestamp: '1970-01-01 00:00:00+00:00'
  verificationStatus: LIBRARY_NO_TESTS
  verifiedWith: []
documentation_of: libs/array_create.py
layout: document
redirect_from:
- /library/libs/array_create.py
- /library/libs/array_create.py.html
title: libs/array_create.py
---
