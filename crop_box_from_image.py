import os
import cv2
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
import traceback

root_images = "./data/file_json/"
root_ouput = "./data/ttsdtb_cropped/"


def order_point(pts):
    rect = np.zeros((4, 2), dtype='float32')
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(img, pts):
    (tl, tr, br, bl) = pts
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    max_width = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_height = max(int(heightA), int(heightB))
    dst = np.array(
        [[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]],
        dtype='float32')
    m = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(img, m, (max_width, max_height))
    return warped


i = 0
for file in glob.glob(root_images + "*"):
    try:
        file_name = os.path.basename(file)
        with open(file, 'r') as f:
            label = json.load(f)

        path = "./data/file_img/"
        file_name = file_name.replace(".json", ".jpg")
        img = cv2.imread(path+file_name)

        if img is None:
            file_name = file_name.replace(".jpg", ".jpeg")
            img = cv2.imread(path + file_name)

        cnt = np.array(label['shapes'][0]['points'], dtype='float32')
        wraped = four_point_transform(img, cnt)
        cv2.imwrite("./data/ttsdtb_cropped/" + file_name, wraped)

    except Exception as e:
        print(e)
        print(traceback.format_exc())

print("done")
    # cv2.imshow("Warped", wraped)
    # cv2.waitKey()
    # print("" + i)

