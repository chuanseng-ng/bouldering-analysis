#!/bin/bash
set -euo pipefail
echo "Will move to src directory to execute main.py"
pushd "$(dirname "${BASH_SOURCE[0]}")/src" > /dev/null
echo "Move done!"
echo ""
echo "Will now execute main.py"
python main.py
popd > /dev/null
echo "Script execution is completed!"
