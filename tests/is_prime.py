# competitive-verifier: PROBLEM https://judge.yosupo.jp/problem/primality_test
from libs.math_func import is_prime
from libs.standard_input import *
from libs.yn_func import *

Q = ii()

while Q:
    N = ii()
    YN(is_prime(N))
    Q -= 1
