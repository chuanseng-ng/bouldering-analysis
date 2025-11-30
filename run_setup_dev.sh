#!/bin/sh -f
# run_setup_dev.csh: C-shell script to run the development setup script

python -m pip install -r requirements.txt
python setup_dev.py
