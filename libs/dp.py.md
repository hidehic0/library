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
  code: "# DP\u306E\u30C6\u30F3\u30D7\u30EC\u30FC\u30C8\nfrom typing import List\n\
    \n\ndef partial_sum_dp(lis: List[int], X: int) -> List[bool]:\n    \"\"\"\n  \
    \  \u90E8\u5206\u548Cdp\u306E\u30C6\u30F3\u30D7\u30EC\u30FC\u30C8\n    lis\u306F\
    \u54C1\u7269\u3067\u3059\n    dp\u914D\u5217\u306E\u9577\u3055\u306F\u3001X\u306B\
    \u3057\u307E\u3059\n    \u8A08\u7B97\u91CF\u306F\u3001O(X*len(L))\u307F\u305F\u3044\
    \u306A\u611F\u3058\n\n    \u8FD4\u308A\u5024\u306F\u3001dp\u914D\u5217\u3067\u4E2D\
    \u8EAB\u306F\u5230\u9054\u3067\u304D\u305F\u304B\u3092\u3001\u793A\u3059bool\u3067\
    \u3059\n    \"\"\"\n    dp = [False] * (X + 1)\n    dp[0] = True\n\n    for a\
    \ in lis:\n        for k in reversed(range(len(dp))):\n            if not dp[k]:\n\
    \                continue\n\n            if k + a >= len(dp):\n              \
    \  continue\n\n            dp[k + a] = True\n\n    return dp\n\n\ndef knapsack_dp(lis:\
    \ List[List[int]], W: int) -> List[int]:\n    \"\"\"\n    \u30CA\u30C3\u30D7\u30B5\
    \u30C3\u30AFdp\u306E\u30C6\u30F3\u30D7\u30EC\u30FC\u30C8\n    lis\u306F\u54C1\u7269\
    \u306E\u30EA\u30B9\u30C8\n    \u539F\u5247\u54C1\u7269\u306F\u3001w,v\u306E\u5F62\
    \u3067\u4E0E\u3048\u3089\u308C\u3001w\u304C\u91CD\u3055\u3001v\u304C\u4FA1\u5024\
    \u3001\u3068\u306A\u308B\n    \u4FA1\u5024\u3068\u91CD\u3055\u3092\u9006\u8EE2\
    \u3055\u305B\u305F\u3044\u5834\u5408\u306F\u81EA\u5206\u3067\u3084\u3063\u3066\
    \u304F\u3060\u3055\u3044\n    dp\u914D\u5217\u306F\u3001\u5B9A\u6570\u500D\u9AD8\
    \u901F\u5316\u306E\u305F\u3081\u3001\u4E00\u6B21\u5143\u914D\u5217\u3068\u3057\
    \u3066\u6271\u3046\n    dp\u914D\u5217\u306E\u9577\u3055\u306F\u3001W\u3068\u3057\
    \u307E\u3059\n    \"\"\"\n\n    dp = [-(1 << 63)] * (W + 1)\n    dp[0] = 0\n\n\
    \    for w, v in lis:\n        for k in reversed(range(len(dp))):\n          \
    \  if w + k >= len(dp):\n                continue\n\n            dp[w + k] = max(dp[w\
    \ + k], dp[k] + v)\n\n    return dp\n\n\ndef article_breakdown(lis: List[List[int]])\
    \ -> List[List[int]]:\n    \"\"\"\n    \u500B\u6570\u5236\u9650\u4ED8\u304D\u30CA\
    \u30C3\u30D7\u30B5\u30C3\u30AF\u306E\u54C1\u7269\u3092\u5206\u89E3\u3057\u307E\
    \u3059\n    \u500B\u6570\u306E\u5024\u304C\u3001\u5404\u54C1\u7269\u306E\u4E00\
    \u756A\u53F3\u306B\u3042\u308C\u3070\u6B63\u5E38\u306B\u52D5\u4F5C\u3057\u307E\
    \u3059\n    \"\"\"\n    res = []\n    for w, v, c in lis:\n        k = 1\n   \
    \     while c > 0:\n            res.append([w * k, v * k])\n            c -= k\n\
    \            k = min(2 * k, c)\n\n    return res\n"
  dependsOn: []
  isVerificationFile: false
  path: libs/dp.py
  requiredBy: []
  timestamp: '1970-01-01 00:00:00+00:00'
  verificationStatus: LIBRARY_NO_TESTS
  verifiedWith: []
documentation_of: libs/dp.py
layout: document
redirect_from:
- /library/libs/dp.py
- /library/libs/dp.py.html
title: libs/dp.py
---
