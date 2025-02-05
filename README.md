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
|素数判定|ミラーラビン素数判定法|$`O(\log N)`$|$`N <= 2^64`$の場合のみ、動作します|
|素数列挙|エラトステネスの篩|$`O(N \log \log N)`$||
|素因数分解|√N以下の数で試し割り|$`O(\sqrt N)`$||
|複数の数を素因数分解|$`\sqrt N`$以下の素数で試し割り|$`O(|a|(\sqrt max(a) \log \log \sqrt max(a)))`$||
|約数列挙|√N以下の数で試し割り|$`O(\sqrt N)`$||
|1からNまでの総和|和の公式|$`O(1)`$|いつも忘れてしまうので関数化した|

