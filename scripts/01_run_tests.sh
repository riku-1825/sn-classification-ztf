source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
run_logged "tests" pytest tests/ -q
