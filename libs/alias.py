# alias
from typing import Any, Iterator, List, Set


def reverserange(*args) -> Iterator:
    """
    rangeをreversedした結果を出力
    返り値はIteratorなので注意して
    """
    return reversed(range(*args))


def listmap(*args) -> List[Any]:
    """
    mapの結果をlist化して出力
    """
    return list(map(*args))


def setmap(*args) -> Set[Any]:
    """
    mapの結果をset化して出力
    """
    return set(map(*args))
