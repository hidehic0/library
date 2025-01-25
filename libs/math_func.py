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
    # 約数全列挙
    import heapq

    result = []

    for i in range(1, N + 1):
        if i * i > N:
            break

        if N % i != 0:
            continue

        heapq.heappush(result, i)
        if N // i != i:
            heapq.heappush(result, N // i)

    return result


def factorization(n):
    # 素因数分解
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


def simple_sigma(n: int) -> int:
    r"""
    1からnまでの総和を求める関数
    つまり和の公式
    """
    return (n * (n + 1)) // 2
