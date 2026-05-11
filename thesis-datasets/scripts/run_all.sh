#!/usr/bin/env bash
set -euo pipefail

python3 thesis-datasets/scripts/build_canonical.py
python3 thesis-datasets/scripts/validate_corpus.py
python3 thesis-datasets/scripts/export_for_sipit.py
python3 thesis-datasets/scripts/export_for_nla.py
