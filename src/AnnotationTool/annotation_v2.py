# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 15:23:38 2024

@author: MaxGr
"""
import os
import cv2
import json

import matplotlib.pyplot as plt
import numpy as np

import math


def draw_overlay(overlay, points):
    """
    draw a set of points on overlay
    Args:
        overlay(numpy array): overlay image
        points(numpy array): a set of three points
    """    
    a,b,c = points
    angle, centroid = calculate_angle(a, b, c)

    text_position = np.array((b[0]-centroid[0], b[1]-centroid[1])).astype(int)  # Above the centroid
    text_position -= [10,0]
    angle_text = f"{angle:.1f}"
    #cv2.putText(temp_image, angle_text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 255), 1)
    
    cv2.putText(overlay, angle_text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 255), 1)
    cv2.circle(overlay, a, 3, (0, 0, 255), -1)
    cv2.circle(overlay, b, 3, (0, 255, 0), -1)
    cv2.circle(overlay, c, 3, (0, 0, 255), -1)
    cv2.line(overlay, a, b, (0, 255, 0), 1)
    cv2.line(overlay, b, c, (0, 255, 0), 1)

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

def delete_clicked_points(x1,y1):
    global annotation,annotation_overlay,temp_image
    cleared = False
    for i,item in enumerate(annotation):
        a,b,c = item['points']
        x2,y2 = b
        search_size = 5
        if (abs(x1-x2) <= search_size) & (abs(y1-y2) <= search_size):
            print(f'found {i} {x2} {y2}')
            annotation.remove(item)
            temp_image = channels[current_channel].copy()
            annotation_overlay.fill(0)
            for item in annotation:
                points = item['points']
                draw_overlay(annotation_overlay, points)
            mask = np.any(annotation_overlay != 0, axis=2)
            temp_image[mask] = annotation_overlay[mask]
            cv2.imshow('overlay', annotation_overlay)
            return
    print('point not found')
    
    
def click_event(event, x, y, flags, param):
    global points, label_done, image, temp_image, angle, centroid
    if event == cv2.EVENT_LBUTTONDOWN:
        display_overlay = True
        point = (x,y)
        points.append(point)
        if len(points) > 3:
            label_done = False
            current_annotation = {}
            points = points[3:]
            temp_image = channels[current_channel].copy()
        if len(points) == 2:
            cv2.circle(temp_image, (x, y), 3, (0, 255, 0), -1) # Green
        else:
            cv2.circle(temp_image, (x, y), 3, (0, 0, 255), -1) # Red
        if len(points) == 3:
            label_done = True
            a,b,c = points
            cv2.line(temp_image, a, b, (0, 255, 0), 1)  
            cv2.line(temp_image, b, c, (0, 255, 0), 1)  
            angle, centroid = calculate_angle(a, b, c)
        
    elif event == cv2.EVENT_RBUTTONDOWN:
        delete_clicked_points(x,y)
    
    cv2.imshow('Image', temp_image)

LEFT_ARROW = 2424832
RIGHT_ARROW = 2555904

annotation = []
label = True
show_original = False
# Example Usage
path = './samples/'
RGB_path = './raw/'
output_json_path = './annotation_json/'
output_image_path = './annotation_images/'
# file = 'eyeQ_RGB.png'

for p in [output_json_path,output_image_path]:
    if not os.path.exists(p):
        os.makedirs(p)

file_list = os.listdir(path)

file_index = 0
count = 0
current_channel = 0

file = file_list[file_index]

cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
cv2.setMouseCallback('Image', click_event)

image = cv2.imread(path + file) #modified image
image_RGB = cv2.imread(RGB_path + file)
image_org = image.copy() #unmodified image
temp_image = image.copy()
annotation_overlay = np.zeros_like(image_RGB)
channels = {0:image, 1:image_RGB}

if file in os.listdir(output_image_path): #continue
    json_file = output_json_path+file+'.json'
    if os.path.exists(json_file):
        with open(json_file, "r") as f:
            temp = json.load(f)
        annotation = temp['annotations']
        for item in annotation:
            points = item['points']
            draw_overlay(annotation_overlay, points)

mask = np.any(annotation_overlay != 0, axis=2)
temp_image[mask] = annotation_overlay[mask]

display_overlay = True

while label:
    if not label: break

    
    index = 0
    points = []
    current_annotation = {}
    label_done = False
    total_angle = 0

    
    cv2.imshow('Image', temp_image)
    cv2.imshow('overlay', annotation_overlay)
    
    key = cv2.waitKeyEx(0)
    print(key)
    
    if key == 9:
        display_overlay ^= True
    if key == LEFT_ARROW:
        file_index = file_index + len(file_list) - 1
        file_index = file_index % len(file_list)
    if key == RIGHT_ARROW:
        file_index = file_index + len(file_list) + 1
        file_index = file_index % len(file_list)
    if key == LEFT_ARROW or key == RIGHT_ARROW:
        file = file_list[file_index]
        print(file)
        image = cv2.imread(path + file)
        image_RGB = cv2.imread(RGB_path + file)
        image_org = image.copy()
        temp_image = image.copy()
        channels = {0:image, 1:image_RGB}
        annotation_overlay.fill(0)
        if file in os.listdir(output_image_path): #continue
            json_file = output_json_path+file+'.json'
            if os.path.exists(json_file):
                with open(json_file, "r") as f:
                    temp = json.load(f)
                annotation = temp['annotations']
                for item in annotation:
                    points = item['points']
                    draw_overlay(annotation_overlay, points)

    if key == ord('s'):
        print(label_done)
        if label_done is True:
            current_annotation = {'points':points, 'angle':angle}
            print(current_annotation)
            annotation.append(current_annotation)
            draw_overlay(annotation_overlay, points)
            #image = temp_image.copy()
            label_done = False
            index += 1
            total_angle += angle
            current_annotation = {}
            points = []
            
    if key == ord('1'):
        current_channel = 0
    if key == ord('2'):
        current_channel = 1

    elif key == 13:  # Enter key
        if len(annotation)==0: 
            print('empty annotation')
            break
        # Convert points to a dictionary for JSON serialization
        annotations_dict = {}
        # for i in annotation:
        #     annotations_dict #???
        output = {
            'file': file,
            'average angle': total_angle/(len(annotation)),
            'annotations': annotation
        }
        cv2.imwrite(output_image_path+file, image)
        with open(f'{output_json_path}{file}.json', 'w') as f:
            json.dump(output, f)
        print(f'annatation save to {output_json_path}{file}.json, {len(annotation)} annatations in total')

    elif key == 27:  # Escape key
        label = False
        break
    
    temp_image = channels[current_channel].copy()
    if display_overlay:
        mask = np.any(annotation_overlay != 0, axis=2)
        temp_image[mask] = annotation_overlay[mask]

        
cv2.destroyAllWindows()
    












