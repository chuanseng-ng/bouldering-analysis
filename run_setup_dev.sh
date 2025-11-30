#!/bin/bash
set -euo pipefail
# run_setup_dev.csh: Bash-shell script to run the development setup script

python -m pip install -r requirements.txt
python setup_dev.py
