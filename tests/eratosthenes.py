# competitive-verifier: PROBLEM https://judge.u-aizu.ac.jp/onlinejudge/description.jsp?id=0009
from libs.math_func import eratosthenes
from libs.standard_input import *

primes = set(eratosthenes(1000000))
acc = [0] * (1000001)

for i in range(1, len(acc)):
    acc[i] = acc[i - 1]
    if i in primes:
        acc[i] += 1

while True:
    try:
        print(acc[ii()])
    except:
        break
