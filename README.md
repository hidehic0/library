# hidehic0's library

[![run unittest](https://github.com/hidehic0/library/actions/workflows/unittest.yml/badge.svg?branch=main)](https://github.com/hidehic0/library/actions/workflows/unittest.yml)
[![CodeQL Advanced](https://github.com/hidehic0/library/actions/workflows/codeql.yml/badge.svg)](https://github.com/hidehic0/library/actions/workflows/codeql.yml)
</br>

[![total lines](https://tokei.rs/b1/github/hidehic0/library)](https://github.com/XAMPPRocky/tokei)
![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/hidehic0/library)
![GitHub repo size](https://img.shields.io/github/repo-size/hidehic0/library)

![GitHub commit week activity](https://img.shields.io/github/commit-activity/w/hidehic0/library)
![GitHub commit month activity](https://img.shields.io/github/commit-activity/m/hidehic0/library)
![GitHub commit year activity](https://img.shields.io/github/commit-activity/y/hidehic0/library)

僕が競プロで使っているライブラリです</br>
主にpythonのライブラリです、c++のもあるけど、いかにも普通です</br>
ファイルは、分割したのは、libsに、全文はcodeに、置いてあります</br>
</br></br>

# 主要な機能

### [数学系](libs/math_func.py)
|機能|アルゴリズム|計算量|備考|
|--|--|--|--|
|素数判定|ミラーラビン素数判定法|$`O(\log N)`$|$`N <= 2^{64}`$の場合のみ、動作します|
|素数列挙|エラトステネスの篩|$`O(N \log \log N)`$||
|素因数分解|√N以下の数で試し割り|$`O(\sqrt N)`$||
|複数の数を素因数分解|$`\sqrt N`$以下の素数で試し割り|$`O(len(a) \times (\sqrt max(a) \log \log \sqrt max(a)))`$||
|約数列挙|√N以下の数で試し割り|$`O(\sqrt N)`$||
|1からNまでの総和|和の公式|$`O(1)`$|いつも忘れてしまうので関数化した|

### [数列作成](libs/array_create.py)
二次元配列、三次元配列を作成できます</br>
ただのスニペットです

### [二分探索](libs/binary_search.py)
二分探索を関数化しただけ

### [ModInt](libs/modint.py)
ModIntのクラス</br>
加算、減算、乗算は、$`O(1)`$</br>
除算は、$`O(\log MOD)`$</br>
べき乗は、$`O(\log b)`$

### [標準入力](libs/standard_input.py)
普通です

### [YesNo関数](libs/yn_func.py)
特色は、YesまたはNoを出力して、exitする関数ぐらいです

### [グリッド関数](libs/grid.py)
* 座標がグリッドの中にあるか判定する関数
* 次に行ける場所を列挙する関数

### [グラフのオブジェクト](libs/grath.py)
スペルミスに気づいた</br>
標準入力で、辺を追加する関数がある

### [UnionFind](libs/unionfind.py)
いつ使うかも、分からないrollback機能がある

### [trie木](libs/trie.py)
いつか使う時が来ると、信じたい
