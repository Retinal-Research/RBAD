# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 15:08:57 2024

@author: MaxGr
"""


import os
import matplotlib.pyplot as plt
import numpy as np
import cv2

import math
import random

import skimage.morphology as skmorph
import copy

from PIL import Image

from collections import deque
import imageio


def img_uint8(image):
    image = (image - np.min(image)) / (np.max(image) - np.min(image)) * 255
    image = image.astype(np.uint8)
    return image

def manhattan_distance(a, b):
    distance = np.sum(np.abs(np.array(a) - np.array(b)))
    return distance

def generate_gaussian_image(height, width, center_point, sigma):
    # Create a grid of coordinates
    (center_x, center_y) = center_point
    x = np.arange(width) - center_x
    y = np.arange(height) - center_y
    x, y = np.meshgrid(x, y)

    # Calculate the squared distances from the center
    distances_squared = x ** 2 + y ** 2

    # Calculate the Gaussian kernel
    kernel = np.exp(-distances_squared / (2 * sigma ** 2))

    # Normalize the kernel to the range [0, 1]
    kernel_normalized = (kernel - np.min(kernel)) / (np.max(kernel) - np.min(kernel))

    return kernel_normalized

def calculate_angle(a, b, c):
    """
    Calculates the absolute angle between line segments ab and bc.

    Args:
        a (tuple): Coordinates of point a (x, y).
        b (tuple): Coordinates of point b (x, y).
        c (tuple): Coordinates of point c (x, y).

    Returns:
        float: The absolute angle in degrees.
    """

    # Calculate vectors ab and bc
    ba_x, ba_y = b[0] - a[0], b[1] - a[1]
    bc_x, bc_y = b[0] - c[0], b[1] - c[1]
    
    centroid = np.array([(ba_x+bc_x)/3, (ba_y+bc_y)/3])

    # Calculate dot product and magnitudes
    dot_product = ba_x * bc_x + ba_y * bc_y
    mag_ab = math.sqrt(ba_x**2 + ba_y**2)
    mag_bc = math.sqrt(bc_x**2 + bc_y**2)

    # Calculate angle in radians
    cos_angle = dot_product / (mag_ab * mag_bc)
    angle_rad = math.acos(cos_angle)

    # Convert to degrees
    angle_deg = math.degrees(angle_rad)

    return angle_deg, centroid


def fast_keypoints(image, roots, tail=15, save_gif=False): 
    '''
    main function for angle detection
    
    input image is a binary skeleton mask
    
    '''
    if save_gif:
        gif_path = './test/angle.gif'
        gif_writer = imageio.get_writer(gif_path, duration=0.001, loop=0)  # Set loop parameter to 0 for infinite loop
        
    plate = copy.deepcopy(image)
    plate = plate.astype(bool)
    
    mask = copy.deepcopy(plate).astype(np.uint8) *255
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) *255
    angle_map = copy.deepcopy(gray)
    
    # plt.imshow(plate)
    # plt.imshow(mask)
    # plt.imshow(gray)

    frame = 0
    count = 0
    keypoints = []
    # angles = []
    center = roots[0]
    point_list = deque()
    for y, x in roots:
        # node, node_type, node_level, step, parent
        point = [(x,y), 0, 0, 0, center]
        point_list.append(point)
        keypoints.append(point)
        cv2.circle(mask, (x, y), 3, (0, 255, 0), -1)
    
    
    while len(point_list)>0:
        frame += 1
        point = point_list[0]
        # print(count, point)
        (dx,dy), node_type, parent_level, step, parent = point
        point_list.popleft()
    
        plate[dy,dx] = 0
        mask[dy,dx] = 0
        gray[dy,dx] = [0,127,0]
        angle_map[dy,dx] = [0,127,0]
    
        boxFrame = plate[dy-1:dy+2, dx-1:dx+2]
        # plt.imshow(plate[temp_y-10:temp_y+11, temp_x-10:temp_x+11])
        # plt.imshow(plate[tempY-50:tempY+51, tempX-50:tempX+51])
        
        num_node = np.sum(boxFrame)
        
        # End node
        if num_node == 0:
            # end_node = [[temp_x,temp_y], 2, parent_level+1, parent]
            end_node = [(dx, dy), 2, parent_level+1, step, parent]
            keypoints.append(end_node)
              
            cv2.circle(mask, (dx,dy), 3, (255, 0, 0), -1)
            # cv2.circle(image_draw, (dx,dy), 3, (255, 0, 0), -1)
            # cv2.arrowedLine(mask, parent, (dx, dy), (255, 0, 0), thickness=2)
    
            count = count+1
            
        elif num_node > 0:
            [Ys,Xs] = np.where(boxFrame > 0)
            exactXs = Xs + dx - 1
            exactYs = Ys + dy - 1
            exact = np.vstack((exactXs,exactYs)).T.tolist()
            
            # Normal node
            if num_node == 1:
                plate[exact[0][1],exact[0][0]] = 0
                new_node = [tuple(exact[0]), 3, parent_level, step+1, parent]
                point_list.append(new_node)
                # point_list.appendleft(new_node)
                
                if step==tail:
                    new_node = [tuple(exact[0]), 4, parent_level, step+1, parent]
                    keypoints.append(new_node)
                    cv2.arrowedLine(angle_map, parent, (dx, dy), (255, 255, 255), thickness=2)
                
            # Bifurcation
            elif num_node > 1:
                bi_node = [(dx, dy), 1, parent_level+1, 0, parent]
                keypoints.append(bi_node)
                for j in range(num_node):
                    plate[exact[j][1],exact[j][0]] = 0
                    new_node = [exact[j], 3, parent_level+1, 0, (dx,dy)]
                    point_list.append(new_node)
                    
                    cv2.circle(mask, (dx,dy), 3, (0, 200, 255), -1)
                    # cv2.circle(image_draw, (dx,dy), 3, (0, 200, 255), -1)
    
                    count = count+1
    
        if frame%50==0 and save_gif:
            result = cv2.add(gray, mask)
            gif_writer.append_data(result)
    
    if save_gif: gif_writer.close()
    # return image_draw, mask, gray, angle_map, keypoints
    return mask, gray, angle_map, keypoints






