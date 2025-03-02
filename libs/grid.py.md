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
  code: "def coordinate_check(x: int, y: int, H: int, W: int) -> bool:\n    \"\"\"\
    \n    \u5EA7\u6A19\u304C\u30B0\u30EA\u30C3\u30C9\u306E\u7BC4\u56F2\u5185\u306B\
    \u3042\u308B\u304B\u30C1\u30A7\u30C3\u30AF\u3059\u308B\u95A2\u6570\n    0-indexed\u304C\
    \u524D\u63D0\n    \"\"\"\n\n    return 0 <= x < H and 0 <= y < W\n\n\nfrom typing\
    \ import List, Tuple\n\n\ndef grid_moves(\n    x: int,\n    y: int,\n    H: int,\n\
    \    W: int,\n    moves: List[Tuple[int]] = [(0, 1), (0, -1), (1, 0), (-1, 0)],\n\
    \    *check_funcs,\n) -> List[Tuple[int]]:\n    \"\"\"\n    \u73FE\u5728\u306E\
    \u5EA7\u6A19\u304B\u3089\u3001\u79FB\u52D5\u53EF\u80FD\u306A\u5EA7\u6A19\u3092\
    moves\u3092\u3082\u3068\u306B\u5217\u6319\u3057\u307E\u3059\u3002\n    x\u3068\
    y\u306F\u73FE\u5728\u306E\u5EA7\u6A19\n    H\u3068W\u306F\u30B0\u30EA\u30C3\u30C9\
    \u306E\u30B5\u30A4\u30BA\n    moves\u306F\u79FB\u52D5\u3059\u308B\u5EA7\u6A19\u304C\
    \u3044\u304F\u3064\u304B\u3092\u4FDD\u5B58\u3059\u308B\n    check_funcs\u306F\u3001\
    \u305D\u306E\u5EA7\u6A19\u306E\u70B9\u304C#\u3060\u3068\u304B\u3092\u81EA\u524D\
    \u3067\u5B9F\u88C5\u3057\u3066\u5224\u5B9A\u306F\u3053\u3061\u3089\u3067\u3059\
    \u308B\u307F\u305F\u3044\u306A\u611F\u3058\n    \u306A\u304Acheck_funcs\u306F\u5F15\
    \u6570\u304Cx\u3068y\u3060\u3051\u3068\u3044\u3046\u306E\u304C\u6761\u4EF6\n \
    \   \u8FFD\u52A0\u306E\u5224\u5B9A\u95A2\u6570\u306F\u3001\u5F3E\u304F\u5834\u5408\
    \u306F\u3001False \u305D\u308C\u4EE5\u5916\u306A\u3089True\u3067\n    \"\"\"\n\
    \    res = []\n\n    for mx, my in moves:\n        nx, ny = x + mx, y + my\n\n\
    \        if not coordinate_check(nx, ny, H, W):\n            continue\n\n    \
    \    for f in check_funcs:\n            if not f(nx, ny):\n                break\n\
    \        else:\n            res.append((nx, ny))\n\n    return res\n"
  dependsOn: []
  isVerificationFile: false
  path: libs/grid.py
  requiredBy: []
  timestamp: '1970-01-01 00:00:00+00:00'
  verificationStatus: LIBRARY_NO_TESTS
  verifiedWith: []
documentation_of: libs/grid.py
layout: document
redirect_from:
- /library/libs/grid.py
- /library/libs/grid.py.html
title: libs/grid.py
---
