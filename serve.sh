#!/bin/bash
python3 pose_estimation.py -k calibration_matrix.npy -d distortion_coefficients.npy  -t DICT_6X6_1000 -p | socat STDIN tcp-l:1234,reuseaddr,fork

