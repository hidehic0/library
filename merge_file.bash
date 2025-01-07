#!/usr/bin/env bash
rm code/main.py
touch code/main.py

# テンプレ
# /bin/cat python/<filename>.py >> code/main.py

cat python/import.py >> code/main.py
cat python/math_func.py >> code/main.py
cat python/array_create.py >> code/main.py
cat python/standard_input.py >> code/main.py
cat python/yn_func.py >> code/main.py
cat python/memo.py >> code/main.py
cat python/grath.py >> code/main.py
cat python/unionfind.py >> code/main.py
cat python/trie.py >> code/main.py
