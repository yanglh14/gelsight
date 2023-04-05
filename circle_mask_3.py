import numpy as np
import cv2
import os
from os import listdir
from os.path import isfile, join, exists
from affine_transform import affine_transform
from marker_mask import *
import torch
from fast_possion import fast_poisson
import matplotlib.pyplot as plt

def get_touch_mask_by_selection(img,circle_center=None,circle_radius=None):

    if circle_center == None or circle_radius == None:

        depth_img = get_depth_image(img)

        (y,x) = np.where(depth_img==depth_img.max())
        y = y[0]
        x = x[0]
        radius = 19

    else:
        x = circle_center[0]
        y = circle_center[1]
        radius = circle_radius

    key = -1

    while key != 32 and key!=27:
        center = (int(x), int(y))
        radius = min(int(radius), 19)

        cirloc_img = cv2.circle(np.array(img), center, int(radius), (0, 40, 0), 2)
        cv2.imshow('img', cirloc_img)
        key = cv2.waitKey(0)
        if key == 119:
            y -= 1
        elif key == 115:
            y += 1
        elif key == 97:
            x -= 1
        elif key == 100:
            x += 1
        elif key == 109:
            radius += 1
        elif key == 110:
            radius -= 1

    cv2.destroyAllWindows()

    # ret,thresh_img = cv2.threshold(depth_img, 160, 255, cv2.THRESH_BINARY)
    #
    # cv2.imshow('thresh_img',thresh_img)
    # cv2.waitKey(0)
    #
    # contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # areas = [cv2.contourArea(c) for c in contours]
    # sorted_areas = np.sort(areas)
    # cnt = contours[areas.index(sorted_areas[-1])]  # the biggest contour
    #
    # (x, y), radius = cv2.minEnclosingCircle(cnt)

    return center, radius

def calibration_data_cropper(data_folder):
    ballfiles = [f for f in listdir(data_folder) if isfile(join(data_folder, f)) and "cal" in f and f.endswith('.jpg')]

    file_pointer = 0

    key = -1
    while(1):
        img_filepath = join(data_folder, ballfiles[file_pointer])
        #print("filepath: " + img_filepath)
        img = affine_transform(cv2.imread(img_filepath))

        pre, ext = os.path.splitext(ballfiles[file_pointer])
        crop_filename = pre + ".txt"
        crop_filepath = join(data_folder+"ball_position/", crop_filename)

        if exists(crop_filepath):
            # read the center and radius
            with open(crop_filepath, 'r+') as f:
                # line: x y radius
                line = f.readline()
                circle_data = line.split()
                x = int(circle_data[0])
                y = int(circle_data[1])
                radius = int(circle_data[2])
                new_center, new_radius = get_touch_mask_by_selection(img,(x,y),radius)
                new_line = str(new_center[0]) + " " + str(new_center[1]) + " " + str(new_radius)
                f.truncate(0)
                f.seek(0)
                f.write(new_line)
        else:
            with open(crop_filepath, "w") as f:
                center, radius = get_touch_mask_by_selection(img)
                line = str(center[0]) + " " + str(center[1]) + " " + str(radius)
                f.write(line)

        #key = cv2.waitKey(0)
        if key == 27:
            break

        file_pointer = (file_pointer + 1) % len(ballfiles)

def infer_gradient(feature):
    model = torch.load('model/model_noxy.pt',map_location=torch.device('cpu'))
    gradient = model(torch.tensor(feature,dtype=torch.float32,device='cpu'))

    return gradient

def get_input(img, mask):

    # calculate the gx and gy
    x = np.linspace(0, img.shape[1] - 1, img.shape[1])
    y = np.linspace(0, img.shape[0] - 1, img.shape[0])
    x, y = np.meshgrid(x, y)

    RGB = img[mask>0]

    X = x[mask>0]
    Y = y[mask>0]

    return RGB,X,Y

def plt_show(img):

    plt.figure()
    plt.imshow(img)
    plt.colorbar(label='Depth')
    plt.title('Depth image')
    plt.xlabel('X Pixel')
    plt.ylabel('Y Pixel')
    plt.show()

def get_depth_image(img):
    dot_mask = get_marker_mask(affine_transform(img))

    touch_mask = np.zeros(img.shape[:2])
    touch_mask[30:300,30:220] = 255

    valid_mask = (1 - dot_mask / 255) * touch_mask

    # infer gradient
    RGB, X, Y = get_input(img, valid_mask)
    feature = np.column_stack((RGB, X, Y))
    gradient = infer_gradient(feature[:,:3])
    gradient = np.array(gradient.tolist())

    gx = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
    gy = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)

    for i in range(feature.shape[0]):
        gx[int(feature[i,4]),int(feature[i,3])] = gradient[i,0]
        gy[int(feature[i,4]),int(feature[i,3])] = gradient[i,1]

    # RECONSTRUCTION
    img_ = fast_poisson(gx,gy)
    img_ -= img_.min()
    img_ = (img_ *255/img_.max()).astype('uint8')
    cv2.imshow('dot',img_)
    cv2.waitKey(0)

    return img_
if __name__ == "__main__":
    calibration_folder = "data/calibration3/"
    ball_ref = "bg-0.jpg"
    pixmm = 0.1  # 0.1mm/pixel
    Rmm = 2.42  # ball radius
    R = Rmm / pixmm

    calibration_data_cropper(calibration_folder)
