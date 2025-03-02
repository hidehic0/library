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
  code: "from typing import List\n\n\n# \u6570\u5B66\u578B\u95A2\u6570\ndef is_prime(n:\
    \ int) -> int:\n    \"\"\"\n    \u7D20\u6570\u5224\u5B9A\u3057\u307E\u3059\n \
    \   \u8A08\u7B97\u91CF\u306F\u5B9A\u6570\u6642\u9593\u3067\u3059\u3002\u6B63\u78BA\
    \u306B\u306F\u3001\u7E70\u308A\u8FD4\u3057\u4E8C\u4E57\u6CD5\u306E\u8A08\u7B97\
    \u91CF\u306B\u3088\u308A\u3067\u3059\n    \u30A2\u30EB\u30B4\u30EA\u30BA\u30E0\
    \u306F\u30DF\u30E9\u30FC\u30E9\u30D3\u30F3\u306E\u7D20\u6570\u5224\u5B9A\u3092\
    \u4F7F\u7528\u3057\u3066\u3044\u307E\u3059\n    n\u304C2^64\u3092\u8D8A\u3048\u308B\
    \u3068\u52D5\u4F5C\u3057\u307E\u305B\u3093\n    \"\"\"\n    if n == 1:\n     \
    \   return False\n\n    def f(a, t, n):\n        x = pow(a, t, n)\n        nt\
    \ = n - 1\n        while t != nt and x != 1 and x != nt:\n            x = pow(x,\
    \ 2, n)\n            t <<= 1\n\n        return t & 1 or x == nt\n\n    if n ==\
    \ 2:\n        return True\n    elif n % 2 == 0:\n        return False\n\n    d\
    \ = n - 1\n    d >>= 1\n\n    while d & 1 == 0:\n        d >>= 1\n\n    checklist\
    \ = (\n        [2, 7, 61] if 2**32 > n else [2, 325, 9375, 28178, 450775, 9780504,\
    \ 1795265022]\n    )\n\n    for i in checklist:\n        if i >= n:\n        \
    \    break\n        if not f(i, d, n):\n            return False\n\n    return\
    \ True\n\n\ndef eratosthenes(n: int) -> List[int]:\n    \"\"\"\n    n\u4EE5\u4E0B\
    \u306E\u7D20\u6570\u3092\u5217\u6319\u3057\u307E\u3059\n    \u8A08\u7B97\u91CF\
    \u306F\u3001O(n log log n)\u3067\u3059\n    \u5148\u7A0B\u306E\u7D20\u6570\u5224\
    \u5B9A\u6CD5\u3067\u5217\u6319\u3059\u308B\u3088\u308A\u3082\u3001\u5C11\u3057\
    \u901F\u3044\u3067\u3059\n    \u5217\u6319\u3057\u305F\u7D20\u6570\u306F\u6607\
    \u9806\u306B\u4E26\u3093\u3067\u3044\u307E\u3059\n    \u30A2\u30EB\u30B4\u30EA\
    \u30BA\u30E0\u306F\u30A8\u30E9\u30C8\u30B9\u30C6\u30CD\u30B9\u3067\u3059\n   \
    \ \"\"\"\n    primes = [True] * (n + 1)\n    primes[0], primes[1] = False, False\n\
    \    i = 2\n    while i**2 <= n:\n        if primes[i]:\n            for k in\
    \ range(i * 2, n + 1, i):\n                primes[k] = False\n\n        i += 1\n\
    \n    return [i for i, p in enumerate(primes) if p]\n\n\ndef calc_divisors(n:\
    \ int):\n    \"\"\"\n    N\u306E\u7D04\u6570\u5217\u6319\u3057\u307E\u3059\n \
    \   \u8A08\u7B97\u91CF\u306F\u3001\u221AN\u3067\u3059\n    \u7D04\u6570\u306F\u6607\
    \u9806\u306B\u4E26\u3093\u3067\u3044\u307E\u3059\n    \"\"\"\n    result = []\n\
    \n    for i in range(1, n + 1):\n        if i * i > n:\n            break\n\n\
    \        if n % i != 0:\n            continue\n\n        result.append(i)\n  \
    \      if n // i != i:\n            result.append(n // i)\n\n    return sorted(result)\n\
    \n\ndef factorization(n: int) -> List[List[int]]:\n    \"\"\"\n    n\u3092\u7D20\
    \u56E0\u6570\u5206\u89E3\u3057\u307E\u3059\n    \u8A08\u7B97\u91CF\u306F\u3001\
    \u221AN\u3067\u3059(\u8981\u6539\u5584)\n    \u8907\u6570\u56DE\u7D20\u56E0\u6570\
    \u5206\u89E3\u3092\u884C\u306A\u3046\u5834\u5408\u306F\u3001\u221AN\u4EE5\u4E0B\
    \u306E\u7D20\u6570\u3092\u5217\u6319\u3057\u305F\u306E\u3067\u8A66\u3057\u5272\
    \u308A\u3057\u305F\u6CD5\u304C\u901F\u3044\u3067\u3059\n    \"\"\"\n    result\
    \ = []\n    tmp = n\n    for i in range(2, int(-(-(n**0.5) // 1)) + 1):\n    \
    \    if tmp % i == 0:\n            cnt = 0\n            while tmp % i == 0:\n\
    \                cnt += 1\n                tmp //= i\n            result.append([i,\
    \ cnt])\n\n    if tmp != 1:\n        result.append([tmp, 1])\n\n    if result\
    \ == []:\n        result.append([n, 1])\n\n    return result\n\n\ndef factorization_plural(L:\
    \ List[int]) -> List[List[List[int]]]:\n    \"\"\"\n    \u8907\u6570\u306E\u6570\
    \u306E\u7D20\u56E0\u6570\u5206\u89E3\u3092\u884C\u306A\u3044\u307E\u3059\n   \
    \ \u8A08\u7B97\u91CF\u306F\u3001O(N * (\u221Amax(L) log log \u221Amax(L)))\n \
    \   \u307F\u305F\u3044\u306A\u611F\u3058\u3067\u3059\n\n    \u6700\u521D\u306B\
    \u7D20\u6570\u3092\u5217\u6319\u3059\u308B\u305F\u3081\u3001\u666E\u901A\u306E\
    \u7D20\u56E0\u6570\u5206\u89E3\u3088\u308A\u52B9\u7387\u304C\u3044\u3044\u3067\
    \u3059\n    \"\"\"\n    res = []\n    primes = eratosthenes(int(max(L) ** 0.5)\
    \ + 20)\n\n    def solve(n):\n        t = []\n        for p in primes:\n     \
    \       if n % p == 0:\n                cnt = 0\n                while n % p ==\
    \ 0:\n                    cnt += 1\n                    n //= p\n\n          \
    \      t.append([p, cnt])\n\n        if n != 1:\n            t.append([n, 1])\n\
    \n        if t == []:\n            t.append([n, 1])\n\n        return t\n\n  \
    \  for n in L:\n        res.append(solve(n))\n\n    return res\n\n\ndef simple_sigma(n:\
    \ int) -> int:\n    \"\"\"\n    1\u304B\u3089n\u307E\u3067\u306E\u7DCF\u548C\u3092\
    \u6C42\u3081\u308B\u95A2\u6570\n    \u3064\u307E\u308A\u548C\u306E\u516C\u5F0F\
    \n    \"\"\"\n    return (n * (n + 1)) // 2\n\n\ndef comb(n: int, r: int, mod:\
    \ int | None = None) -> int:\n    \"\"\"\n    \u9AD8\u901F\u306A\u306F\u305A\u306E\
    \u4E8C\u9805\u4FC2\u6570\n    mod\u3092\u6307\u5B9A\u3059\u308C\u3070\u3001mod\u4ED8\
    \u304D\u306B\u306A\u308B\n    \"\"\"\n    a = 1\n\n    for i in range(n - r +\
    \ 1, n + 1):\n        a *= i\n\n        if mod:\n            a %= mod\n\n    b\
    \ = 1\n\n    for i in range(1, r + 1):\n        b *= i\n        if mod:\n    \
    \        b %= mod\n\n    if mod:\n        return a * pow(b, -1, mod) % mod\n \
    \   else:\n        return a * b\n"
  dependsOn: []
  isVerificationFile: false
  path: libs/math_func.py
  requiredBy: []
  timestamp: '1970-01-01 00:00:00+00:00'
  verificationStatus: LIBRARY_NO_TESTS
  verifiedWith: []
documentation_of: libs/math_func.py
layout: document
redirect_from:
- /library/libs/math_func.py
- /library/libs/math_func.py.html
title: libs/math_func.py
---
