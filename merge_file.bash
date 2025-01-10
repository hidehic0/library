#!/bin/bash
rm code/main.py
touch code/main.py

echo "新しいmain.py作成完了"

# テンプレ
# /bin/cat python/<filename>.py >> code/main.py

cat python/import.py >> code/main.py
cat python/math_func.py >> code/main.py
cat python/array_create.py >> code/main.py
cat python/binary_search.py >> code/main.py
cat python/standard_input.py >> code/main.py
cat python/yn_func.py >> code/main.py
cat python/grid.py >> code/main.py
cat python/memo.py >> code/main.py
cat python/grath.py >> code/main.py
cat python/unionfind.py >> code/main.py
cat python/trie.py >> code/main.py
cat python/utils.py >> code/main.py

echo "作業完了"

cat code/main.py
