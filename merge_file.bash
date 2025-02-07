#!/bin/bash
rm code/main.py
touch code/main.py

echo "新しいmain.py作成完了"

# テンプレ
# /bin/cat python/<filename>.py >> code/main.py

cat libs/import.py >>code/main.py
cat libs/math_func.py >>code/main.py
cat libs/array_create.py >>code/main.py
cat libs/binary_search.py >>code/main.py
cat libs/modint.py >>code/main.py
cat libs/standard_input.py >>code/main.py
cat libs/yn_func.py >>code/main.py
cat libs/grid.py >>code/main.py
cat libs/dp.py >>code/main.py
cat libs/memo.py >>code/main.py
cat libs/grath.py >>code/main.py
cat libs/unionfind.py >>code/main.py
cat libs/trie.py >>code/main.py
cat libs/bit.py >>code/main.py
cat libs/dis.py >>code/main.py
cat libs/utils.py >>code/main.py

echo "作業完了"

cat code/main.py
