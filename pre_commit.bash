#!/bin/bash

./merge_file.bash
ruff check
ruff format
git add -A
