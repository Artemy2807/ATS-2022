# coding: utf-8
import cv2
import numpy as np


def binarize(img, d=0):
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    binary_h = cv2.inRange(hls, (0, 0, 10), (255, 255, 255))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary_g = cv2.inRange(gray, 220, 255)

    binary = cv2.bitwise_and(binary_g, binary_h)
    kernel = np.ones((2, 2), 'uint8')
    binary_dilate = cv2.dilate(binary, kernel, iterations=1)

    if d:
        cv2.imshow('hls', hls)
        cv2.imshow('hlsRange', binary_h)
        cv2.imshow('grayRange', binary_g)
        cv2.imshow('bin_dilate', binary_dilate)
        cv2.imshow('bin', binary)

    # return binary
    return binary_dilate


def binarize_exp(img, d=0):
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hls_s_channel = hls[:, :, 2]
    hls_l_channel = hls[:, :, 1]
    hls_h_channel = hls[:, :, 0]
    hsv_h_channel = hsv[:, :, 2]
    hsv_s_channel = hsv[:, :, 1]
    hsv_v_channel = hsv[:, :, 0]
    binary_h = cv2.inRange(hls, (0, 0, 30), (255, 255, 205))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary_g = cv2.inRange(gray, 130, 255) #130
    binary = cv2.bitwise_and(binary_g, binary_h)

    if d:
        cv2.imshow('hls', hls)
        cv2.imshow('bgr_b', img[:, :, 0])
        cv2.imshow('bgr_g', img[:, :, 1])
        cv2.imshow('bgr_r', img[:, :, 2])
        cv2.imshow('hls_s', hls_s_channel)
        cv2.imshow('hls_l', hls_l_channel)
        cv2.imshow('hls_h', hls_h_channel)
        cv2.imshow('hsv_h', hsv_h_channel)
        cv2.imshow('hsv_s', hsv_s_channel)
        cv2.imshow('hsv_v', hsv_v_channel)
        cv2.imshow('hlsRange', binary_h)
        cv2.imshow('grayRange', binary_g)
        cv2.imshow('gray', gray)
        cv2.imshow('bin', binary)

    # return binary
    return binary

def trans_perspective(binary, trap, rect, size, d=0):
    matrix_trans = cv2.getPerspectiveTransform(trap, rect)
    perspective = cv2.warpPerspective(binary, matrix_trans, size, flags=cv2.INTER_LINEAR)
    #if d:
        #cv2.imshow('perspective', perspective)
    return perspective

def detect_distance_mark(perspective):
    distance_mark = 0
    for i in range(100, 180):
        #print(int(np.sum(perspective[i, :], axis=0) // 255))
        distance_mark += int(np.sum(perspective[i, :], axis=0) // 255)
    #print(distance_mark)
    return distance_mark >= 3000


def find_left_right(perspective, d=0):
    hist = np.sum(perspective[perspective.shape[0] // 3:, :], axis=0)
    mid = hist.shape[0] // 2
    left = np.argmax(hist[:mid])
    right = np.argmax(hist[mid:]) + mid
    if left <= 10 and right - mid <= 10:
        right = 399

    if d:
        cv2.line(perspective, (left, 0), (left, 300), 50, 2)
        cv2.line(perspective, (right, 0), (right, 300), 50, 2)
        cv2.line(perspective, ((left + right) // 2, 0), ((left + right) // 2, 300), 110, 3)
        cv2.imshow('lines', perspective)

    return left, right


def centre_mass(perspective, d=0):
    hist = np.sum(perspective, axis=0)
    if d:
        cv2.imshow("Perspektiv2in", perspective)

    mid = hist.shape[0] // 2
    i = 0
    centre = 0
    sum_mass = 0
    while (i <= mid):
        centre += hist[i] * (i + 1)
        sum_mass += hist[i]
        i += 1
    if sum_mass > 0:
        mid_mass_left = centre / sum_mass
    else:
        mid_mass_left = mid-1
    if (sum_mass // 255) < 1000:
        mid_mass_left = mid

    centre = 0
    sum_mass = 0
    i = mid
    while i < hist.shape[0]:
        centre += hist[i] * (i + 1)
        sum_mass += hist[i]
        i += 1
    if sum_mass > 0:
        mid_mass_right = centre / sum_mass
    else:
        mid_mass_right = mid+1
    if (sum_mass // 255) < 1000:
        mid_mass_right = mid

    # print(mid_mass_left)
    # print(mid_mass_right)
    mid_mass_left = int(mid_mass_left)
    mid_mass_right = int(mid_mass_right)
    if d:
        cv2.line(perspective, (mid_mass_left, 0), (mid_mass_left, perspective.shape[1]), 50, 2)
        cv2.line(perspective, (mid_mass_right, 0), (mid_mass_right, perspective.shape[1]), 50, 2)
        cv2.line(perspective, ((mid_mass_right + mid_mass_left) // 2, 0), ((mid_mass_right + mid_mass_left) // 2, perspective.shape[1]), 110, 3)
        cv2.imshow('CentrMass', perspective)

    return mid_mass_left, mid_mass_right


def detect_stop(perspective):
    hist = np.sum(perspective, axis=1)
    maxStrInd = np.argmax(hist)
    # print("WhitePixCual" + str(hist[maxStrInd]//255))
    if hist[maxStrInd]//255 > 150:
        # print("SL detected. WhitePixselCual: "+str(int(hist[maxStrInd]/255)) + "Ind: " + str(maxStrInd))
        if maxStrInd > 120:  # 100
            # print("Time to stop")
            # cv2.line(perspective, (0, maxStrInd), (perspective.shape[1], maxStrInd), 60, 4)
            # cv2.imshow("STOP| ind:"+str(maxStrInd)+"IndCual"+str(hist[maxStrInd]//255), perspective)
            return True
    return False


def detect_road_begin(perspective):  # для переключения с пересеченипея перекрёстка на следование по разметке
    left_corner = np.sum(perspective[-50:, :perspective.shape[1] // 3])
    right_corner = np.sum(perspective[-50:, perspective.shape[1] // 3 * 2:])
    print(left_corner)
    print(right_corner)
    print("**----**")
    # if left_corner >= 500000 and right_corner >= 170000:
    # if left_corner >= 170000 and right_corner >= 170000:
    if left_corner >= 170000 and right_corner >= 170000:
        return True
    else:
        return False
