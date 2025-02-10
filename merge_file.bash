#!/bin/bash
rm code/main.py
touch code/main.py

echo "新しいmain.py作成完了"

# テンプレ
# /bin/cat python/<filename>.py >> code/main.py

function add_code() {
  local lib_path
  lib_path="libs/${1}"
  cat $lib_path >>code/main.py
}

add_code "import.py"
add_code "math_func.py"
add_code "array_create.py"
add_code "binary_search.py"
add_code "modint.py"
add_code "standard_input.py"
add_code "yn_func.py"
add_code "grid.py"
add_code "dp.py"
add_code "memo.py"
add_code "grath.py"
add_code "unionfind.py"
add_code "trie.py"
add_code "bit.py"
add_code "dis.py"
add_code "utils.py"

echo "作業完了"

cat code/main.py
