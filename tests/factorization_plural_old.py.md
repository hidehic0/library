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
  code: "# python tests/factorization_plural_old.py  42.42s user 0.00s system 99%\
    \ cpu 42.515 total\n\n\ndef factorization(n):\n    \"\"\"\n    n\u3092\u7D20\u56E0\
    \u6570\u5206\u89E3\u3057\u307E\u3059\n    \u8A08\u7B97\u91CF\u306F\u3001\u221A\
    N\u3067\u3059(\u8981\u6539\u5584)\n    \u8907\u6570\u56DE\u7D20\u56E0\u6570\u5206\
    \u89E3\u3092\u884C\u306A\u3046\u5834\u5408\u306F\u3001\u221AN\u4EE5\u4E0B\u306E\
    \u7D20\u6570\u3092\u5217\u6319\u3057\u305F\u306E\u3067\u8A66\u3057\u5272\u308A\
    \u3057\u305F\u6CD5\u304C\u901F\u3044\u3067\u3059\n    \"\"\"\n    result = []\n\
    \    tmp = n\n    for i in range(2, int(-(-(n**0.5) // 1)) + 1):\n        if tmp\
    \ % i == 0:\n            cnt = 0\n            while tmp % i == 0:\n          \
    \      cnt += 1\n                tmp //= i\n            result.append([i, cnt])\n\
    \n    if tmp != 1:\n        result.append([tmp, 1])\n\n    if result == []:\n\
    \        result.append([n, 1])\n\n    return result\n\n\nt = [10**10] * (10**4)\n\
    \nfor n in t:\n    factorization(n)\n"
  dependsOn: []
  isVerificationFile: false
  path: tests/factorization_plural_old.py
  requiredBy: []
  timestamp: '2025-03-02 19:35:59+09:00'
  verificationStatus: LIBRARY_NO_TESTS
  verifiedWith: []
documentation_of: tests/factorization_plural_old.py
layout: document
redirect_from:
- /library/tests/factorization_plural_old.py
- /library/tests/factorization_plural_old.py.html
title: tests/factorization_plural_old.py
---
