#!/bin/bash
rm code/main.py
touch code/main.py

# テンプレ
# /bin/cat python/<filename>.py >> code/main.py

/bin/cat python/import.py >> code/main.py
/bin/cat python/math_func.py >> code/main.py
/bin/cat python/array_create.py >> code/main.py
/bin/cat python/standard_input.py >> code/main.py
/bin/cat python/yn_func.py >> code/main.py
/bin/cat python/memo.py >> code/main.py
/bin/cat python/grath.py >> code/main.py
/bin/cat python/unionfind.py >> code/main.py
/bin/cat python/trie.py >> code/main.py
