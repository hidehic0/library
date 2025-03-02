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
  code: "# python tests/factorization_plural_new.py  2.58s user 0.01s system 99% cpu\
    \ 2.591 total\n\n\ndef eratosthenes(n):\n    \"\"\"\n    n\u4EE5\u4E0B\u306E\u7D20\
    \u6570\u3092\u5217\u6319\u3057\u307E\u3059\n    \u8A08\u7B97\u91CF\u306F\u3001\
    O(n log log n)\u3067\u3059\n    \u5148\u7A0B\u306E\u7D20\u6570\u5224\u5B9A\u6CD5\
    \u3067\u5217\u6319\u3059\u308B\u3088\u308A\u3082\u3001\u5C11\u3057\u901F\u3044\
    \u3067\u3059\n    \u5217\u6319\u3057\u305F\u7D20\u6570\u306F\u6607\u9806\u306B\
    \u4E26\u3093\u3067\u3044\u307E\u3059\n    \u30A2\u30EB\u30B4\u30EA\u30BA\u30E0\
    \u306F\u30A8\u30E9\u30C8\u30B9\u30C6\u30CD\u30B9\u3067\u3059\n    \"\"\"\n   \
    \ primes = [True] * (n + 1)\n    primes[0], primes[1] = False, False\n    i =\
    \ 2\n    while i**2 <= n:\n        if primes[i]:\n            for k in range(i\
    \ * 2, n + 1, i):\n                primes[k] = False\n\n        i += 1\n\n   \
    \ return [i for i, p in enumerate(primes) if p]\n\n\ndef factorization_plural(L):\n\
    \    \"\"\"\n    \u8907\u6570\u306E\u6570\u306E\u7D20\u56E0\u6570\u5206\u89E3\u3092\
    \u884C\u306A\u3044\u307E\u3059\n    \u8A08\u7B97\u91CF\u306F\u3001O(N * (\u221A\
    max(L) log log \u221Amax(L)))\n    \u307F\u305F\u3044\u306A\u611F\u3058\u3067\u3059\
    \n\n    \u6700\u521D\u306B\u7D20\u6570\u3092\u5217\u6319\u3059\u308B\u305F\u3081\
    \u3001\u666E\u901A\u306E\u7D20\u56E0\u6570\u5206\u89E3\u3088\u308A\u52B9\u7387\
    \u304C\u3044\u3044\u3067\u3059\n    \"\"\"\n    res = []\n    primes = eratosthenes(int(max(L)\
    \ ** 0.5) + 20)\n\n    def solve(n):\n        t = []\n        for p in primes:\n\
    \            if n % p == 0:\n                cnt = 0\n                while n\
    \ % p == 0:\n                    cnt += 1\n                    n //= p\n\n   \
    \             t.append([p, cnt])\n\n        if n != 1:\n            t.append([n,\
    \ 1])\n\n        if t == []:\n            t.append([n, 1])\n\n        return t\n\
    \n    for n in L:\n        res.append(solve(n))\n\n    return res\n\n\nt = [10**10]\
    \ * (10**4)\n\nfactorization_plural(t)\n"
  dependsOn: []
  isVerificationFile: false
  path: tests/factorization_plural_new.py
  requiredBy: []
  timestamp: '2025-03-02 19:35:59+09:00'
  verificationStatus: LIBRARY_NO_TESTS
  verifiedWith: []
documentation_of: tests/factorization_plural_new.py
layout: document
redirect_from:
- /library/tests/factorization_plural_new.py
- /library/tests/factorization_plural_new.py.html
title: tests/factorization_plural_new.py
---
