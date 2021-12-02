#!/bin/bash
python3 pose_estimation.py "$@" | socat STDIN tcp-l:1234,reuseaddr,fork
