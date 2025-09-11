#!/bin/bash

./merge_file.bash
ruff check --exclude tests --exclude libs/import.py --exclude code/main.py
ruff format
git add -A
