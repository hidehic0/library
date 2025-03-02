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
  code: "# YesNo\u95A2\u6570\ndef YesNoTemplate(state: bool, upper: bool = False)\
    \ -> str:\n    \"\"\"\n    state\u304CTrue\u306A\u3089\u3001upper\u306B\u5FDC\u3058\
    \u3066Yes,YES\u3092return\n    state\u304CFalse\u306A\u3089\u3001upper\u306B\u5FDC\
    \u3058\u3066No,NO\u3092return\u3059\u308B\n    \"\"\"\n    YES = [\"Yes\", \"\
    YES\"]\n    NO = [\"No\", \"NO\"]\n\n    if state:\n        return YES[int(upper)]\n\
    \    else:\n        return NO[int(upper)]\n\n\ndef YN(state: bool, upper: bool\
    \ = False) -> None:\n    \"\"\"\n    \u5148\u7A0B\u306EYesNoTemplate\u95A2\u6570\
    \u306E\u7D50\u679C\u3092\u51FA\u529B\u3059\u308B\n    \"\"\"\n    res = YesNoTemplate(state,\
    \ upper)\n\n    print(res)\n\n\ndef YE(state: bool, upper: bool = False) -> bool\
    \ | None:\n    \"\"\"\n    bool\u304CTrue\u306A\u3089Yes\u3092\u51FA\u529B\u3057\
    \u3066exit\n    \"\"\"\n\n    if not state:\n        return False\n\n    YN(True,\
    \ upper)\n    exit()\n\n\ndef NE(state: bool, upper: bool = False) -> bool | None:\n\
    \    \"\"\"\n    bool\u304CTrue\u306A\u3089No\u3092\u51FA\u529B\u3057\u3066exit\n\
    \    \"\"\"\n\n    if not state:\n        return False\n\n    YN(False, upper)\n\
    \    exit()\n"
  dependsOn: []
  isVerificationFile: false
  path: libs/yn_func.py
  requiredBy: []
  timestamp: '2025-03-02 19:35:59+09:00'
  verificationStatus: LIBRARY_NO_TESTS
  verifiedWith: []
documentation_of: libs/yn_func.py
layout: document
redirect_from:
- /library/libs/yn_func.py
- /library/libs/yn_func.py.html
title: libs/yn_func.py
---
