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
  code: "# \u6A19\u6E96\u5165\u529B\u95A2\u6570\nimport sys\nfrom typing import Any,\
    \ List\n\n\ndef s() -> str:\n    \"\"\"\n    \u4E00\u884C\u306B\u4E00\u3064\u306E\
    string\u3092input\n    \"\"\"\n    return sys.stdin.readline().rstrip()\n\n\n\
    def sl() -> List[str]:\n    \"\"\"\n    \u4E00\u884C\u306B\u8907\u6570\u306Estring\u3092\
    input\n    \"\"\"\n    return s().split()\n\n\ndef ii() -> int:\n    \"\"\"\n\
    \    \u4E00\u3064\u306Eint\n    \"\"\"\n    return int(s())\n\n\ndef il(add_num:\
    \ int = 0) -> List[int]:\n    \"\"\"\n    \u4E00\u884C\u306B\u8907\u6570\u306E\
    int\n    \"\"\"\n    return list(map(lambda i: int(i) + add_num, sl()))\n\n\n\
    def li(n: int, func, *args) -> List[List[Any]]:\n    \"\"\"\n    \u8907\u6570\u884C\
    \u306E\u5165\u529B\u3092\u30B5\u30DD\u30FC\u30C8\n    \"\"\"\n    return [func(*args)\
    \ for _ in [0] * n]\n"
  dependsOn: []
  isVerificationFile: false
  path: libs/standard_input.py
  requiredBy: []
  timestamp: '1970-01-01 00:00:00+00:00'
  verificationStatus: LIBRARY_NO_TESTS
  verifiedWith: []
documentation_of: libs/standard_input.py
layout: document
redirect_from:
- /library/libs/standard_input.py
- /library/libs/standard_input.py.html
title: libs/standard_input.py
---
