#!/bin/bash
rm code/main.py
touch code/main.py

# テンプレ
# /bin/cat python/<filename>.py >> code/main.py

/bin/cat python/import.py >> code/main.py
/bin/cat python/math_func.py >> code/main.py
/bin/cat python/array_create.py >> code/main.py
