# python tests/factorization_plural_old.py  42.42s user 0.00s system 99% cpu 42.515 total


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


t = [10**10] * (10**4)

for n in t:
    factorization(n)
