# 標準入力関数
import sys


# 一行に一つのstring
def s():
    return sys.stdin.readline().rstrip()


# 一行に複数のstring
def sl():
    return s().split()


# 一つのint
def ii():
    return int(s())


# 一行に複数のint
def il(add_num: int = 0):
    return list(map(lambda i: int(i) + add_num, sl()))


# 複数行の入力をサポート
def li(n: int, func, *args):
    return [func(*args) for _ in [0] * n]
