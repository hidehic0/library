# Trie木
class Trie:
    class Data:
        def __init__(self, value, ind):
            self.count = 1
            self.value = value
            self.childs = {}
            self.ind = ind

    def __init__(self):
        self.data = [self.Data("ab", 0)]  # 初期値はabにして被らないようにする

    def add(self, value: str) -> int:
        cur = 0
        result = 0

        # 再帰的に探索する
        for t in value:
            childs = self.data[cur].childs  # 参照渡しで

            if t in childs:
                self.data[childs[t]].count += 1
            else:
                nd = self.Data(t, len(self.data))
                childs[t] = len(self.data)
                self.data.append(nd)

            result += self.data[childs[t]].count - 1
            cur = childs[t]

        return result

    def lcp_max(self, value: str) -> int:
        cur = 0
        result = 0

        for t in value:
            childs = self.data[cur].childs

            if t not in childs:
                break

            if self.data[childs[t]].count == 1:
                break

            cur = childs[t]
            result += 1

        return result

    def lcp_sum(self, value: str) -> int:
        cur = 0
        result = 0

        for t in value:
            childs = self.data[cur].childs

            if t not in childs:
                break

            if self.data[childs[t]].count == 1:
                break

            cur = childs[t]
            result += self.data[childs[t]].count - 1

        return result
