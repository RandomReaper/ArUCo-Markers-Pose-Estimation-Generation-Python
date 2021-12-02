#!/usr/bin/env python3
'''
This program will estimate pose of ArUCo markers
  * Sample usage : ./pose_estimation.py
  * Help         : ./pose_esitmation.py -h
'''
import numpy as np
import cv2
import sys
from utils import ARUCO_DICT
import argparse
import time
import math
import stat
import os

font = cv2.FONT_HERSHEY_SIMPLEX
#--- 180 deg rotation matrix around the x axis
R_flip  = np.zeros((3,3), dtype=np.float32)
R_flip[0,0] = 1.0
R_flip[1,1] =-1.0
R_flip[2,2] =-1.0

#------------------------------------------------------------------------------
# Rotations functions where kindly borrowed from :
# https://www.learnopencv.com/rotation-matrix-to-euler-angles/
#------------------------------------------------------------------------------
def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R):
    assert (isRotationMatrix(R))

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])

#------------------------------------------------------------------------------
# End of code borrow
#------------------------------------------------------------------------------


def pose_esitmation(frame, aruco_dict_type, matrix_coefficients, distortion_coefficients, marker_size_mm):

    '''
    frame - Frame from the video stream
    matrix_coefficients - Intrinsic matrix of the calibrated camera
    distortion_coefficients - Distortion coefficients associated with your camera
    marker_size_mm - The size of a marker

    return:-
    frame - The frame with some info on it
    '''
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    corners, ids, rejected_img_points = cv2.aruco.detectMarkers(gray, cv2.aruco_dict,parameters=parameters,
        cameraMatrix=matrix_coefficients,
        distCoeff=distortion_coefficients)

    # If markers are detected
    if len(corners) > 0:
        for i in range(0, len(ids)):
            # Estimate pose of each marker and return the values rvec and tvec---(different from those of camera coefficients)
            rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[i], marker_size_mm, matrix_coefficients,
                                                                       distortion_coefficients)
            # Draw a square around the markers
            if args['preview']:
                cv2.aruco.drawDetectedMarkers(frame, corners)

            # Draw Axis
            if args['preview']:
                cv2.aruco.drawAxis(frame, matrix_coefficients, distortion_coefficients, rvec, tvec, 10)

            if args['preview']:
                x=int(corners[i].reshape((4, 2))[0][0])
                y=int(corners[i].reshape((4, 2))[0][1])
                cv2.putText(frame, str(ids[i])[1:-1],
                    (x, y - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 2)

            #-- Obtain the rotation matrix tag->camera
            R_ct    = np.matrix(cv2.Rodrigues(rvec)[0])
            R_tc    = R_ct.T

            #-- Get the attitude in terms of euler 321 (Needs to be flipped first)
            roll_marker, pitch_marker, yaw_marker = rotationMatrixToEulerAngles(R_flip*R_tc)

            yaw=math.degrees(yaw_marker)
            pitch=math.degrees(pitch_marker)
            roll=math.degrees(roll_marker)

            (x,y,z) = tvec.reshape(3,1)
            x = x[0]
            y = y[0]
            z = z[0]

            print(f"id:{int(ids[i]):03d},"
                    +f"x:{x:+08.3f},"
                    +f"y:{y:+08.3f},"
                    +f"z:{z:+08.3f},"
                    +f"a:{yaw:+08.3f}"
                , flush=True)

            if ids[i] == 3 and args['preview']:
                cv2.putText(frame, "x:{}".format(int(x)),
                    (0, 20),
                    font,
                    0.5, (0, 255, 0), 2)
                cv2.putText(frame, "y:{}".format(int(y)),
                    (0, 40),
                    font,
                    0.5, (0, 255, 0), 2)
                cv2.putText(frame, "z:{}".format(int(z)),
                    (0, 60),
                    font,
                    0.5, (0, 255, 0), 2)

                cv2.putText(frame, "yaw:{}".format(int(yaw)),
                    (0, 80),
                    font,
                    0.5, (0, 255, 0), 2)

                cv2.putText(frame, "pitch:{}".format(int(pitch)),
                    (0, 100),
                    font,
                    0.5, (0, 255, 0), 2)

                cv2.putText(frame, "roll:{}".format(int(roll)),
                    (0, 120),
                    font,
                    0.5, (0, 255, 0), 2)
    else:
        print("NO_MARKER")
    return frame

if __name__ == '__main__':

    # Manage program parameters
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("-d", "--device", type=str, default="/dev/video0", help="name of the video device or filename of a video")
    ap.add_argument("-k", "--K_Matrix", type=str, default="calibration_matrix.npy", help="Path to calibration matrix (numpy file)")
    ap.add_argument("-c", "--D_Coeff", type=str, default="distortion_coefficients.npy", help="Path to distortion coefficients (numpy file)")
    ap.add_argument('-m', "--marker_size_mm", type=int, default=20, help="Size of one side of a maker in mm")
    ap.add_argument('-f', "--framerate", type=int, default=10, help="Limit the processed frame rate (does not change the camera parameters), ignored when using a video file")
    ap.add_argument("-t", "--type", type=str, default="DICT_6X6_1000", help="Type of ArUCo tag to detect")
    ap.add_argument('-p', "--preview", action='store_true', help="Show a preview window")
    ap.add_argument('-x', "--focus", type=int, default=50, help="cv2.CAP_PROP_FOCUS")
    ap.add_argument('-r', "--resolution", nargs=2, type=int, default=(800,600), help="resolution")

    args = vars(ap.parse_args())

    if ARUCO_DICT.get(args["type"], None) is None:
        print(f"ArUCo tag type '{args['type']}' is not supported")
        sys.exit(0)

    aruco_dict_type = ARUCO_DICT[args["type"]]
    calibration_matrix_path = args["K_Matrix"]
    distortion_coefficients_path = args["D_Coeff"]
    marker_size_mm = args["marker_size_mm"]
    frame_rate = args['framerate']
    video_file = args['device']

    # Load calibratino
    k = np.load(calibration_matrix_path)
    d = np.load(distortion_coefficients_path)

    device = False
    # Open the video device/file
    if (stat.S_ISCHR(os.lstat(video_file)[stat.ST_MODE])):
        device = True

    if device:
        video = cv2.VideoCapture(video_file, cv2.CAP_V4L)
    else:
        video = cv2.VideoCapture(video_file)

    if not video.isOpened():
        sys.exit(f"can't open video (file/device): '{video_file}'")

    # Setup video parameters
    if device:
        video.set(cv2.CAP_PROP_FRAME_WIDTH, args['resolution'][0])
        video.set(cv2.CAP_PROP_FRAME_HEIGHT, args['resolution'][1])
        video.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        video.set(cv2.CAP_PROP_FOCUS, args['focus'])

    # Aruco options
    cv2.aruco_dict = cv2.aruco.Dictionary_get(aruco_dict_type)
    parameters = cv2.aruco.DetectorParameters_create()
    parameters.detectInvertedMarker = 0

    if (args['preview']):
        cv2.namedWindow('pose', cv2.WINDOW_NORMAL)

    # Do the work
    prev = 0
    while True:
        now = time.time()
        time_elapsed = now - prev
        ret = video.grab()

        if not ret:
            break

        # Maybe drop some frame when using a live video
        if time_elapsed > 1./frame_rate or not device:
            prev = now
            ret, frame = video.retrieve()
            if not ret:
                break

            output = pose_esitmation(frame, aruco_dict_type, k, d, marker_size_mm)

            if (args['preview']):
                cv2.imshow('pose', output)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

    video.release()
    cv2.destroyAllWindows()
