import sys
import numpy as np
import time
import imutils
import cv2

cv_file = cv2.FileStorage()
cv_file.open('MatrizCalibracion.xml', cv2.FileStorage_READ)
stereoMapL_x = cv_file.getNode('stereoMapL_x').mat()
stereoMapL_y = cv_file.getNode('stereoMapL_y').mat()
stereoMapR_x = cv_file.getNode('stereoMapR_x').mat()
stereoMapR_y = cv_file.getNode('stereoMapR_y').mat()

# Parámetros de distorción y rectificación de la imagen
# Código base obtenido de
# https://github.com/niconielsen32/ComputerVision/blob/master/StereoVisionDepthEstimation/calibration.py
def undistortRectify(frameR, frameL):
    # Se rectifica las imágenes
    undistortedL= cv2.remap(frameL, stereoMapL_x, stereoMapL_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
    undistortedR= cv2.remap(frameR, stereoMapR_x, stereoMapR_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)

    return undistortedR, undistortedL


# Calcula la distancia de la cámara a un punto
# Código base obtenido de
# https://github.com/niconielsen32/ComputerVision/blob/master/StereoVisionDepthEstimation/triangulation.py
def find_depth(right_point, left_point, frame_right, frame_left, baseline,f, alpha):
    
    height_right, width_right= frame_right.shape
    height_left, width_left = frame_left.shape

    if width_right == width_left:
        f_pixel = (width_right * 0.5) / np.tan(alpha * 0.5 * np.pi/180)
    else:
        print('Los frames de las cámaras izquierda y derecha no tienen el mismo tamaño')

    x_right = right_point[0]
    x_left = left_point[0]

    # Cálculo de disparidad entre frame de cámara izquierda y derecha en píxeles
    disparity = x_left-x_right

    # Cálculo de distancia en cm
    zDepth = (baseline*f_pixel)/disparity

    return zDepth

