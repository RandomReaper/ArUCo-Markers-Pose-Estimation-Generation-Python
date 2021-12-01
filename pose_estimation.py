'''
Sample Usage:-
python pose_estimation.py --K_Matrix calibration_matrix.npy --D_Coeff distortion_coefficients.npy --type DICT_5X5_100
'''


import numpy as np
import cv2
import sys
from utils import ARUCO_DICT
import argparse
import time
import math

font = cv2.FONT_HERSHEY_SIMPLEX
#--- 180 deg rotation matrix around the x axis
R_flip  = np.zeros((3,3), dtype=np.float32)
R_flip[0,0] = 1.0
R_flip[1,1] =-1.0
R_flip[2,2] =-1.0

marker_size_mm = 20

#------------------------------------------------------------------------------
#------- ROTATIONS https://www.learnopencv.com/rotation-matrix-to-euler-angles/
#------------------------------------------------------------------------------
# Checks if a matrix is a valid rotation matrix.
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

def pose_esitmation(frame, aruco_dict_type, matrix_coefficients, distortion_coefficients):

    '''
    frame - Frame from the video stream
    matrix_coefficients - Intrinsic matrix of the calibrated camera
    distortion_coefficients - Distortion coefficients associated with your camera

    return:-
    frame - The frame with the axis drawn on it
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

    ap = argparse.ArgumentParser()
    ap.add_argument("-k", "--K_Matrix", required=True, help="Path to calibration matrix (numpy file)")
    ap.add_argument("-d", "--D_Coeff", required=True, help="Path to distortion coefficients (numpy file)")
    ap.add_argument("-t", "--type", type=str, default="DICT_ARUCO_ORIGINAL", help="Type of ArUCo tag to detect")
    ap.add_argument('-p', "--preview", action='store_true')
    ap.add_argument('-f', "--framerate", type=int, default=10)
    args = vars(ap.parse_args())

    if ARUCO_DICT.get(args["type"], None) is None:
        print(f"ArUCo tag type '{args['type']}' is not supported")
        sys.exit(0)

    aruco_dict_type = ARUCO_DICT[args["type"]]
    calibration_matrix_path = args["K_Matrix"]
    distortion_coefficients_path = args["D_Coeff"]

    k = np.load(calibration_matrix_path)
    d = np.load(distortion_coefficients_path)

    video = cv2.VideoCapture(1)
    video.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
    video.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
    video.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    video.set(cv2.CAP_PROP_FOCUS, 50)

    cv2.aruco_dict = cv2.aruco.Dictionary_get(aruco_dict_type)
    parameters = cv2.aruco.DetectorParameters_create()
    parameters.detectInvertedMarker = 1

    frame_rate = args['framerate']
    print(f"frame_rate:{frame_rate}")
    prev = 0

    while True:
        time_elapsed = time.time() - prev
        ret, frame = video.read()

        if not ret:
            break

        if time_elapsed > 1./frame_rate:
            prev = time.time()

            output = pose_esitmation(frame, aruco_dict_type, k, d)

            if (args['preview']):
                cv2.imshow('Estimated Pose', output)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
        else:
            print("drop")

    video.release()
    cv2.destroyAllWindows()
