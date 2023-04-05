import numpy as np
import cv2
import os
from os import listdir
from os.path import isfile, join, exists
from affine_transform import affine_transform
from marker_mask import *

def get_touch_mask(img_touch, center, radius):
    touch_mask = np.zeros((img_touch.shape[0], img_touch.shape[1]), dtype=int)
    touch_mask = cv2.circle(np.array(touch_mask), (int(center[0]), int(center[1])), int(radius), (255, 255, 255), -1)

    return touch_mask

def get_touch_mask_by_selection(img,ref):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)

    dot_mask = get_marker_mask(img, threshold = 100)

    dot_mask_ref = get_marker_mask(ref, threshold = 100)

    valid_mask = (1 - dot_mask / 255) * (1 - dot_mask_ref / 255)

    # cv2.imshow('dot',dot_mask)
    # cv2.waitKey(0)
    # cv2.imshow('dot_ref',dot_mask_ref)
    # cv2.waitKey(0)

    dot_mask = valid_mask<1
    gray[dot_mask] = 0
    gray_ref[dot_mask] = 0

    # cv2.imshow('gray2',gray)
    # cv2.waitKey(0)
    # cv2.imshow('gray_ref',gray_ref)
    # cv2.waitKey(0)

    diff = abs(gray.astype('float')-gray_ref.astype('float'))

    # cv2.imshow('diff',diff.astype('uint8'))
    # cv2.waitKey(0)

    ret,thresh_img = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)

    # cv2.imshow('diff',thresh_img)
    # cv2.waitKey(0)

    contours, hierarchy = cv2.findContours(thresh_img.astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in contours]
    index = np.where(np.array(areas)>5)[0]
    cnt = contours[index[0]]
    for i in range(len(index)-1):
        cnt = np.concatenate([cnt,contours[index[i+1]]],axis=0)

    (x, y), radius = cv2.minEnclosingCircle(cnt)

    #create an empty image for contours
    img_contours = np.zeros(img.shape)

    cv2.circle(img, (int(x), int(y)), int(radius), (0, 255, 255), 2)

    # draw the contours on the empty image
    cv2.drawContours(img_contours, cnt, -1, (0,255,0), 3)
    cv2.imshow('contours',img)
    cv2.waitKey(0)

    return (x, y), radius

def calibration_data_cropper(data_folder, ref_file, ball_radius_p):
    ballfiles = [f for f in listdir(data_folder) if isfile(join(data_folder, f)) and "cal" in f and f.endswith('.jpg')]

    img_bg_filepath = join(data_folder, ref_file)
    img_bg = affine_transform(cv2.imread(img_bg_filepath))
    file_pointer = 0

    ref = cv2.imread('data/calibration/bg-0.jpg')

    key = -1
    while(1):
        img_filepath = join(data_folder, ballfiles[file_pointer])
        #print("filepath: " + img_filepath)
        img = affine_transform(cv2.imread(img_filepath))

        pre, ext = os.path.splitext(ballfiles[file_pointer])
        crop_filename = pre + ".txt"
        crop_filepath = join(data_folder+"ball_position/", crop_filename)


        center, radius = get_touch_mask_by_selection(img, affine_transform(ref))

        #key = cv2.waitKey(0)
        if key == 27:
            break

        file_pointer = (file_pointer + 1) % len(ballfiles)


if __name__ == "__main__":
    calibration_folder = "data/calibration2/"
    ball_ref = "bg-0.jpg"
    pixmm = 0.1  # 0.1mm/pixel
    Rmm = 2.42  # ball radius
    R = Rmm / pixmm

    calibration_data_cropper(calibration_folder, ball_ref, R)
