from typing import List


# 数学型関数
def is_prime(n):
    """
    素数判定します
    計算量は定数時間です。正確には、繰り返し二乗法の計算量によりです
    アルゴリズムはミラーラビンの素数判定を使用しています
    nが2^64を越えると動作しません
    """
    if n == 1:
        return False

    def f(a, t, n):
        x = pow(a, t, n)
        nt = n - 1
        while t != nt and x != 1 and x != nt:
            x = pow(x, 2, n)
            t <<= 1

        return t & 1 or x == nt

    if n == 2:
        return True
    elif n % 2 == 0:
        return False

    d = n - 1
    d >>= 1

    while d & 1 == 0:
        d >>= 1

    checklist = (
        [2, 7, 61] if 2**32 > n else [2, 325, 9375, 28178, 450775, 9780504, 1795265022]
    )

    for i in checklist:
        if i >= n:
            break
        if not f(i, d, n):
            return False

    return True


def eratosthenes(n):
    """
    n以下の素数を列挙します
    計算量は、O(n log log n)です
    先程の素数判定法で列挙するよりも、少し速いです
    列挙した素数は昇順に並んでいます
    アルゴリズムはエラトステネスです
    """
    primes = [True] * (n + 1)
    primes[0], primes[1] = False, False
    i = 2
    while i**2 <= n:
        if primes[i]:
            for k in range(i * 2, n + 1, i):
                primes[k] = False

        i += 1

    return [i for i, p in enumerate(primes) if p]


def calc_divisors(N):
    """
    Nの約数列挙します
    計算量は、√Nです
    約数は昇順に並んでいます
    """
    result = []

    for i in range(1, N + 1):
        if i * i > N:
            break

        if N % i != 0:
            continue

        result.append(i)
        if N // i != i:
            result.append(N // i)

    return sorted(result)


def factorization(n):
    """
    nを素因数分解します
    計算量は、√Nです(要改善)
    複数回素因数分解を行なう場合は、√N以下の素数を列挙したので試し割りした法が速いです
    """
    result = []
    tmp = n
    for i in range(2, int(-(-(n**0.5) // 1)) + 1):
        if tmp % i == 0:
            cnt = 0
            while tmp % i == 0:
                cnt += 1
                tmp //= i
            result.append([i, cnt])

    if tmp != 1:
        result.append([tmp, 1])

    if result == []:
        result.append([n, 1])

    return result


def factorization_plural(L: List[int]) -> List[List[int]]:
    """
    複数の数の素因数分解を行ないます
    計算量は、O(N * (√max(L) log log √max(L)))
    みたいな感じです

    最初に素数を列挙するため、普通の素因数分解より効率がいいです
    """
    res = []
    primes = eratosthenes(int(max(L) ** 0.5) + 20)

    def solve(n):
        t = []
        for p in primes:
            if n % p == 0:
                cnt = 0
                while n % p == 0:
                    cnt += 1
                    n //= p

                t.append([p, cnt])

        if n != 1:
            t.append([n, 1])

        if t == []:
            t.append([n, 1])

        return t

    for n in L:
        res.append(solve(n))

    return res


def simple_sigma(n: int) -> int:
    """
    1からnまでの総和を求める関数
    つまり和の公式
    """
    return (n * (n + 1)) // 2
