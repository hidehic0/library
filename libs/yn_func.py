# YesNo関数
def YesNoTemplate(state: bool, upper: bool = False) -> str:
    """YesNo関数のテンプレート

    stateがTrueなら、upperに応じてYes,YESをreturn
    stateがFalseなら、upperに応じてNo,NOをreturnする
    """
    YES = ["Yes", "YES"]
    NO = ["No", "NO"]

    if state:
        return YES[int(upper)]
    else:
        return NO[int(upper)]


def YN(state: bool, upper: bool = False) -> None:
    """
    先程のYesNoTemplate関数の結果を出力する
    """
    res = YesNoTemplate(state, upper)

    print(res)


def YE(state: bool, upper: bool = False) -> bool | None:
    """
    boolがTrueならYesを出力してexit
    """

    if not state:
        return False

    YN(True, upper)
    exit()


def NE(state: bool, upper: bool = False) -> bool | None:
    """
    boolがTrueならNoを出力してexit
    """

    if not state:
        return False

    YN(False, upper)
    exit()
