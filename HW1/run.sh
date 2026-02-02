#!/bin/bash

set -eou pipefail

uv run main.py --delta_t 0.1
uv run main.py --delta_t 0.01

