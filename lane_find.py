import pygetwindow
import time
import os
import pyautogui
from PIL import ImageGrab, ImageShow, Image
import cv2
import numpy as np
from sklearn.cluster import KMeans
from FastLine import Line
from math import atan2, degrees, sqrt

def create_gaborfilter():
    # from https://www.freedomvc.com/index.php/2021/10/16/gabor-filter-in-edge-detection/
    #Function is designed to produce a set of GaborFilters
    #an even distribution of theta values equally distributed amongst pi rad / 180 degrees.

    filters = []
    num_filters = 4
    ksize = 35
    sigma = 3.0
    lambd = 10.0
    gamma = 0.5
    psi = 0
    for theta in np.arange(0, np.pi, np.pi / num_filters):
        kern = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_64F)
        kern /= 1.0 * kern.sum() #Brightness normalization
        filters.append(kern)
    return filters

def roi(img, vertices, show=0):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked = cv2.bitwise_and(img, mask)
    if show == 1:
        for i in vertices[0]:
            masked = cv2.circle(masked, (i[0], i[1]), 7, (0,255, 0), -1)
    return masked

def apply_filter(img, filters):
    #General function to apply filters to our image
    #New Image
    newimage = np.zeros_like(img)

    #Starting with blank, loop through and apply gabor filt
    #on each it, take highest value until have max
    #final image is returned
    depth = -1
    for kern in filters: #Loop through the kernels in GaborFilter
        image_filter = cv2.filter2D(img, depth, kern)
        #Using numpy maximum to comp filter and cumulative image taking max
        np.maximum(newimage, image_filter, newimage)
    return newimage

#Pass in a numpy array and get x1 y1 x2 y2 values for the
def get_mean_line(lines):
    try:
        x1    = int(lines[:,0].mean())
        y1    = int(lines[:,1].mean())
        x2    = int(lines[:,2].mean())
        y2    = int(lines[:,3].mean())
        slope = degrees(atan2(y2-y1, x2-x1))
    except:
        #Return all 0's as an error handling. Line will just plot as a useless dot.
        print("Array is empty, returning Zeros")
        x1 = 0
        y1 = 0
        x2 = 0
        y2 = 0
        slope = 0
    return x1, y1, x2, y2, slope

def get_target_point(slope):
    #Function to take in the angles and find the difference from 90 degrees and take the two most vertical lines as the Lanes
    lines = np.array([])
    for i in slope:
        if lines.shape[0] == 0:
            angle = i[4]
            lines = np.array([abs(angle)])
        else:
            angle = i[4]
            lines = np.vstack((lines, np.array([abs(angle)])))
    #now use argmax to return indices of the max value
    ind_1big = np.argmax(lines)
    #grab line1
    line1 = slope[ind_1big]
    #set found index to zero, find next biggest... Maybe add some logic here to see the distance of the two lines and if they are close ignore the second line and move on. 
    lines[ind_1big] = 0
    #Save midpoint of line
    x1_mid = (line1[0]+ line1[2])/2
    y1_mid = (line1[1]+ line1[3])/2
    while True:
        ind_2big = np.argmax(lines)
        line2 = slope[ind_2big]
        lines[ind_2big] = 0   #Set index to zero so next man up gets tried.
        #Get midpoint of 2nd line
        x2_mid = (line2[0]+ line2[2])/2
        y2_mid = (line2[1]+ line2[3])/2

        #Get length of line
        dist_midpoints = sqrt((x2_mid- x1_mid)**2 + (y2_mid - y1_mid)**2)
        if dist_midpoints > 100:
            break

    line2 = slope[ind_2big]
    #Calculate line intersection after getting the two lines.
    l1 = Line(p1=(line1[0],line1[1]), p2=(line1[2], line1[3]))
    l2 = Line(p1=(line2[0],line2[1]), p2=(line2[2], line2[3]))

    p = l1.intersection(l2)
    #Here from the two lines, choose the lowest point on each to match with the target for the lane plotting.
    if(line1[3] > line1[1]):
        lane1 = np.array([line1[2],line1[3], p[0], p[1]])
    else:
        lane1 = np.array([line1[0],line1[1], p[0], p[1]])

    if(line2[3] > line2[1]):
        lane2 = np.array([line2[2],line2[3], p[0], p[1]])
    else:
        lane2 = np.array([line2[0],line2[1], p[0], p[1]])
    target_x = p[0]
    target_y = p[1]
    #Returning Target point and the two lanes.
    return target_x, target_y, lane1, lane2

#Here is where we will build the lane finding code on just the picture before it becomes implemented with the game
img = cv2.imread("C:\RoadDetect\Pictures\StructuredRoad\sroad_25.png")

def lane_find(img):
    kernel = np.ones((5,5), np.uint8)
    erode_kernel = np.ones((3,3), np.uint8)
    img_size = img.shape #(row, column) (j, i)

    image_verts = np.array([[int(img_size[1]*0.1), int(img_size[0]*0.7)], [int(img_size[1]*0.1), int(img_size[0]*0.4)], [int(img_size[1]*0.35), int(img_size[0]* 0.2)], [int(img_size[1]*0.65), int(img_size[0]*0.2)], [int(img_size[1]*0.9), int(img_size[0]*0.4)], [int(img_size[1]*0.9), int(img_size[0]*0.7)]])

    #Printing out important region of the image.
    cv2.circle(img, (int(img_size[1]*0.5),int(img_size[0]*0.68)), 4, (250,150,0), -1)
    cv2.circle(img, (int(img_size[1]*0.5),int(img_size[0]*0.4)),  4, (250, 150,0), -1)
    cv2.line(img, (int(img_size[1]*0.5),int(img_size[0]*0.4)), (int(img_size[1]*0.5),int(img_size[0]*0.68)), (0, 255, 255), lineType=cv2.LINE_4, thickness=3)
    #cv2.rectangle(img, (int(img_size[1]*0.15), int(img_size[0]*0.2)), (int(img_size[1]*0.85), int(img_size[0]*0.7)), (150, 250,0), thickness=1)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.erode(gray, erode_kernel)
    gray = cv2.dilate(gray, kernel, iterations=1)
    edges = cv2.Canny(gray, threshold1=50, threshold2=120, apertureSize=3)
    edges = roi(edges, [image_verts], show=0)
    edges  = cv2.GaussianBlur(edges, (5, 5), cv2.BORDER_DEFAULT)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 255, minLineLength=200, maxLineGap=20)

    line_store = np.array([])
    for i in lines:
        for x1,y1,x2,y2 in i:
            slope = (y2-y1)/(x2-x1)
            #cv2.line(img, (x1, y1), (x2, y2), (255,0, 0))
            if line_store.shape[0] is 0:
                line_store = np.array([x1, y1, x2, y2, slope])
                
            else:
                line_store = np.vstack((line_store, np.array([x1, y1, x2, y2, slope])))

    #Grab a cluster based purely on the slope of the lines.
    kmeans = KMeans(n_clusters=6, random_state=0).fit(line_store[:,4].reshape(-1,1))

    #update line_store with the 
    kmean_label = kmeans.labels_
    kmean_label = kmean_label.reshape(-1,1)
    line_store = np.hstack((line_store, kmean_label))

    #Print out the different line groups and end points for an idea of whats going on
    line_0 = np.array([])
    line_1 = np.array([])
    line_2 = np.array([])
    line_3 = np.array([])
    line_4 = np.array([])
    line_5 = np.array([])
    for i in line_store:
        x1 = int(i[0])
        y1 = int(i[1])
        x2 = int(i[2])
        y2 = int(i[3])
        slope = i[4]
        group = i[5]
        if group == 0:
            #cv2.line(img, (x1,y1),(x2,y2),(255,0,0),4)
            if line_0.shape[0] == 0:
                line_0 = np.array([x1, y1, x2, y2, slope, group])
            else:
                line_0 = np.vstack((line_0, [x1, y1, x2, y2, slope, group]))
        elif group == 1:
            #cv2.line(img, (x1,y1),(x2,y2),(0,255,0),4)
            if line_1.shape[0] == 0:
                line_1 = np.array([x1, y1, x2, y2, slope, group])
            else:
                line_1 = np.vstack((line_1, [x1, y1, x2, y2, slope, group]))
        elif group == 2:
            #cv2.line(img, (x1,y1),(x2,y2),(0,0,255),4)
            if line_2.shape[0] == 0:
                line_2 = np.array([x1, y1, x2, y2, slope, group])
            else:
                line_2 = np.vstack((line_2, [x1, y1, x2, y2, slope, group]))
        elif group == 3:
            #cv2.line(img, (x1,y1),(x2,y2),(255,125,0),4)
            if line_3.shape[0] == 0:
                line_3 = np.array([x1, y1, x2, y2, slope, group])
            else:
                line_3 = np.vstack((line_3, [x1, y1, x2, y2, slope, group]))
        elif group == 4:
            #cv2.line(img, (x1,y1),(x2,y2),(255,125,0),4)
            if line_4.shape[0] == 0:
                line_4 = np.array([x1, y1, x2, y2, slope, group])
            else:
                line_4 = np.vstack((line_4, [x1, y1, x2, y2, slope, group]))
        elif group == 5:
            #cv2.line(img, (x1,y1),(x2,y2),(255,125,0),4)
            if line_5.shape[0] == 0:
                line_5 = np.array([x1, y1, x2, y2, slope, group])
            else:
                line_5 = np.vstack((line_5, [x1, y1, x2, y2, slope, group]))
        else:
            cv2.line(img, (x1,y1),(x2,y2),(0,125,255),4)

    #Now get the Line points and slope for plotting
    l0_x1, l0_y1, l0_x2, l0_y2, l0_slope = get_mean_line(line_0)
    l1_x1, l1_y1, l1_x2, l1_y2, l1_slope = get_mean_line(line_1)
    l2_x1, l2_y1, l2_x2, l2_y2, l2_slope = get_mean_line(line_2)
    l3_x1, l3_y1, l3_x2, l3_y2, l3_slope = get_mean_line(line_3)
    l4_x1, l4_y1, l4_x2, l4_y2, l4_slope = get_mean_line(line_4)
    l5_x1, l5_y1, l5_x2, l5_y2, l5_slope = get_mean_line(line_5)

    #Overwrite line info as this is all we need
    line_0 = [l0_x1, l0_y1, l0_x2, l0_y2, l0_slope]
    line_1 = [l1_x1, l1_y1, l1_x2, l1_y2, l1_slope]
    line_2 = [l2_x1, l2_y1, l2_x2, l2_y2, l2_slope]
    line_3 = [l3_x1, l3_y1, l3_x2, l3_y2, l3_slope]
    line_4 = [l4_x1, l4_y1, l4_x2, l4_y2, l4_slope]
    line_5 = [l5_x1, l5_y1, l5_x2, l5_y2, l5_slope]

    #Setup the line array and find the best target point
    lines = np.array([line_0, line_1, line_2, line_3, line_4, line_5])
    x,y,lane1,lane2 = get_target_point(lines)

    #Plot the lanes chosen
    #cv2.line(img, (int(lane1[0]),int(lane1[1])), (int(lane1[2]),int(lane1[3])), (255,150,0),thickness = 10)
    #cv2.line(img, (int(lane2[0]),int(lane2[1])), (int(lane2[2]),int(lane2[3])), (150,255,0),thickness = 10)
    
    #Plot the lines out
    #cv2.line(img, (l0_x1, l0_y1), (l0_x2, l0_y2), (0,255,0), thickness=2)
    #cv2.line(img, (l1_x1, l1_y1), (l1_x2, l1_y2), (255,0,0), thickness=2)
    #cv2.line(img, (l2_x1, l2_y1), (l2_x2, l2_y2), (0,0,255), thickness=2)
    #cv2.line(img, (l3_x1, l3_y1), (l3_x2, l3_y2), (255,0,255), thickness=2)
    #cv2.circle(img, (int(x),int(y)), 7, (255,0,0), -1)

    return x,y,lane1,lane2