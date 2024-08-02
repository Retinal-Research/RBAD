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
            cv2.circle(image_draw, (dx,dy), 3, (255, 0, 0), -1)
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
                    cv2.circle(image_draw, (dx,dy), 3, (0, 200, 255), -1)
    
                    count = count+1
    
        if frame%50==0 and save_gif:
            result = cv2.add(gray, mask)
            gif_writer.append_data(result)
    
    if save_gif: gif_writer.close()
    return image_draw, mask, gray, angle_map, keypoints



from collections import deque
import imageio


path = './DRIVE/training/1st_manual/'

file_list = sorted(os.listdir(path))

RAND = {}

for file_name in file_list:
    
    # file_name = file_list[17]
    file = path + file_name
    print(file)
    
    if file.split('.')[-1] == 'gif':
        image = Image.open(file)
        image = np.array(image)
    else:
        image = cv2.imread(file, 0)
    
    if len(image.shape)==2:
        height,width = image.shape
        
    # image_draw = copy.deepcopy(image).astype(np.uint8) *255
    image_draw = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    # plt.imshow(image_draw)
    
    image_draw[:,:,0] = 0
    image_draw[:,:,1] = image_draw[:,:,1] * 127
    image_draw[:,:,2] = 0
    
    angle_draw = copy.deepcopy(image_draw)
    
    image_skeleton = skmorph.skeletonize(image.astype(bool)).astype(np.uint8)
    plt.imshow(image_skeleton)
    
    
    
    '''
    # Initialize
    '''
    
    
    image_init = copy.deepcopy(image_skeleton)
    plt.imshow(image_init)
    
    roots = np.transpose(np.nonzero(image_init))
    roots_index = np.random.choice(np.arange(len(roots)))
    roots = roots[roots_index]
    
    
    image_draw, mask, plate, angle_map, keypoints = fast_keypoints(image_init, [roots], save_gif=False)
    
    attention_list = []
    for p in keypoints:
        # print(p)
        node, node_type, parent_level, step, parent = p
    
        attention_map = generate_gaussian_image(height, width, parent, 21)
        attention_list.append(attention_map)
        
        if node_type==1:
            color = (0,255,0) 
        elif node_type==2:
            color = (255,255,0)
        else:
            color = (0,255,255) 
            
        cv2.arrowedLine(mask, parent, node, color, thickness=2)
        
    
    
    plt.imshow(mask)
    
    attention_map = np.sum(attention_list, 0).astype(np.uint8)
    plt.imshow(attention_map)
    
    # cv2.imwrite(f'./test/plot/{file_name}_attention_map.png', cv2.cvtColor(attention_map, cv2.COLOR_GRAY2RGB))
    
    
    gradient_map = copy.deepcopy(attention_map)
    
    # Find the index of the maximum value
    center = np.unravel_index(np.argmax(gradient_map), gradient_map.shape)
    
    max_region = np.transpose(np.where(gradient_map==np.max(gradient_map)))
    
    center = np.mean(max_region, 0).astype(int)
    center = tuple([center[1], center[0]])
    
    region_size = 50
    gradient_flow = cv2.applyColorMap(img_uint8(gradient_map), cv2.COLORMAP_PLASMA)
    plt.imshow(gradient_flow)
    
    
    
    
    
    
    
    
    '''
    # Start from the true root
    '''
        
    if file_name == '24_manual1.gif':
        center = (470, 283)
    if file_name == '36_manual1.gif':
        center = (482, 290)
    if file_name == '38_manual1.gif':
        center = (505, 282)
    
    cv2.circle(gradient_flow, center, region_size, (0,0,255), 10)
    cv2.imwrite(f'./test/plot/{file_name}_attention_map.png', gradient_flow)
        
    
    r = 40
    
    image_circle = copy.deepcopy(image_skeleton)
    # plt.imshow(image_circle)
    y_coords, x_coords = np.nonzero(image_circle)

    # np.mgrid[0:image_circle.shape[0], 0:image_circle.shape[1]]
    distances = np.sqrt((x_coords - center[0])**2 + (y_coords - center[1])**2)
    closest_pixel_index = np.unravel_index(np.argmin(distances), distances.shape)
    
    roots = tuple([y_coords[closest_pixel_index], x_coords[closest_pixel_index]])

    
    image_draw, mask, plate, angle_map, keypoints = fast_keypoints(image_circle, [roots], tail=15, save_gif=False)
    
    
    result = cv2.add(plate, mask)

    plt.imshow(result)
    cv2.imwrite(f'./test/plot/{file_name}_keypoints.png', cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    
    plt.imshow(image_draw)
    cv2.imwrite(f'./test/plot/{file_name}_keypoints_draw.png', cv2.cvtColor(image_draw, cv2.COLOR_BGR2RGB))
    
    plt.imshow(angle_map)
    cv2.imwrite(f'./test/plot/{file_name}_angle_map.png', cv2.cvtColor(angle_map, cv2.COLOR_BGR2RGB))
    
    
    Tree = []
    index = 0
    for p in keypoints:
        # print(p)
        node, node_type, parent_level, step, parent = p
        
        child = []
        child_index = []
        for node_i in keypoints:
            if node_i[-1] == node:
                child.append(node_i[0])
                child_index.append(keypoints.index(node_i))
               
        Tree.append([index, node, node_type, parent_level, step, parent, child, child_index])
        index += 1
    
    
    
    
    # angle_draw = copy.deepcopy(plate)
    angle_list = []
    new_bifurcations = []
    for p in Tree:
        # print(p)
        index, node, node_type, parent_level, step, parent, child_raw, child_index = p
        
        child = copy.deepcopy(child_raw)
        if len(child)<2: continue
        # print(child)
        del_list = []
        for i in range(len(child)):
            child_i = Tree[child_index[i]]
            # print(child_i)
            # if child_i[2] != 3: continue
            dist = manhattan_distance(node, child_i[1])
            if dist<5: 
                del_list.append(i)
        
        for index in sorted(del_list, reverse=True):  # Iterate in reverse order
            child.pop(index)
    
        if len(child)<2: continue
        child_a = child[0]
        child_c = child[1]
        
        subnode_distance = manhattan_distance(child_a, child_c)
        if subnode_distance<5: continue
            
        a,b,c = child_a, node, child_c
        # print(a,b,c)
        angle_i,_ = calculate_angle(a,b,c)
        # if angle_i<20: continue
        # if angle_i<20 or 100<angle_i: continue
        if angle_i<20 or 120<angle_i: continue

        angle_list.append([angle_i, child_a, node, child_c])
        
        cv2.arrowedLine(angle_draw, b, a, (255, 255, 255), thickness=2)
        cv2.arrowedLine(angle_draw, b, c, (255, 255, 255), thickness=2)
    
        bifurcation = {
                'points': np.array(node).tolist(),
                'angle': angle_i
                }
        new_bifurcations.append(bifurcation)

    
    plt.imshow(angle_draw)
    cv2.imwrite(f'./test/plot/{file_name}_angle_draw.png', cv2.cvtColor(angle_draw, cv2.COLOR_BGR2RGB))
    
    avg_angle = np.mean(np.array(angle_list, dtype=object)[:,0])
    print(avg_angle)
    
    RAND[file_name] = new_bifurcations






import json
from datetime import datetime

current_datetime = datetime.now()
date_time_string = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")

with open(f'./test/ours_{date_time_string}.json', 'w') as f:
    json.dump(RAND, f)






