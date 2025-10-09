class ChangeMin:
    def __init__(self, x) -> None:
        """Change min構造体

        代入時現在の値より代入する値が低ければ代入される
        setメソッドで代入する
        """
        self.x = x

    def set(self, new) -> None:
        self.x = min(self.x, new)

    def val(self) -> any:
        return self.x


class ChangeMax:
    def __init__(self, x) -> None:
        """Change min構造体

        代入時現在の値より代入する値が大きければ代入される
        setメソッドで代入する
        """
        self.x = x

    def set(self, new) -> None:
        self.x = max(self.x, new)

    def val(self) -> any:
        return self.x
