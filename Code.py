##FIX 1. Length and width are arbitary

# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

#Read me:
#1. Prepare file name to be "condition-no-xk", k = magnification
#2. Put everything in "test_directory" --> LINE 41
#3 specify folder name --> Line 61
#3. Select size range --> Line 143
#4. Run 1 photo detection (first cell) everytime you change directory --> Change photo name --> Line 324
#5. Change 'Not Processed' and 'Processd' images --> Line 299 and 324 
#6.
#5. c = crop more
#6. d = delete that photo --> transfer to another folder to be processed later
    # change directory to keep discarded photo --> Line 295
#7. f x 2= finish
#8. Run multiple photo detection 
    #change directory -> Line 338
    # change number of photo to detect --> Line 342
    # change output talble --> Line 359 (make sure to chage the Table name becuase it will be overided)
# %%
import cv2 as cv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math 
import tkinter as tk
from tkinter import *
import PIL
from PIL import Image, ImageTk
from pynput.mouse import Listener as MouseListener
from pynput.keyboard import Listener as KeyboardListener
import time
import datetime
import argparse
import os as os 

def BGR_to_RGB(image):
    b, g, r = cv.split(image)
    image = cv.merge([r,g,b])
    return image
def gray_to_RGB(image):
    return cv.merge([image, image, image])
test_directory = "K:/Kanit/Postdoc at VISTEC/Project_size selection of MXene/Results/Python code/cascade 20220305/"
test_folders = os.listdir(test_directory)
test_folders.sort()

''' to modify file name '''
### 0min-1-1k.tif --> remove tif ###

def find_file_path(file_name):
    file_name_short =  " "
    if file_name.endswith(("png", "PNG", "tif")): 
        file_name_short = file_name_short + file_name[:-4]
    elif  file_name.endswith(("tiff", "TIFF", "jpeg", "JPEG")):
        file_name_short =file_name_short + file_name[:-5]
    else:
        return "Pls check file format"
    
###''' bounding the position of the 'magnification' '''###
        
    pos1, pos2, pos3 = [i for i, n in enumerate(file_name_short) if n == '-'][-1], file_name_short.find("k"), [i for i, n in enumerate(file_name_short) if n == '-'][0]
    folder_name = file_name_short[1:pos3]
    #folder_name = 'test'
    scale = (int(file_name_short[pos1+1:pos2])/3)*32
    file_path = test_directory+'/'+ folder_name+'/'+ file_name ## specify folder name 
    return file_path, file_name_short, scale

''' collect color value '''

def colors(pic,pixelpoints):
    all_red, all_green, all_blue, all_color = [], [], [], []
    for pixel in pixelpoints:         
        all_red.append      ( pic[pixel[0]][pixel[1]] [0])
        all_green.append    ( pic[pixel[0]][pixel[1]] [1])
        all_blue.append     ( pic[pixel[0]][pixel[1]] [2])
        all_color.append((pic[pixel[0]][pixel[1]][0],pic[pixel[0]][pixel[1]][1],pic[pixel[0]][pixel[1]][2]))
            
    mean_color = (round(np.mean(all_red)),round(np.mean(all_green)),round(np.mean(all_blue)))
    min_color = (min(all_red),min(all_green),min(all_blue))
    max_color = (max(all_red),max(all_green),max(all_blue))
    SD_color = (round(np.std(all_red)),round(np.std(all_green)),round(np.std(all_blue)))
    return mean_color, min_color, max_color, SD_color

''' box of flakes'''

def minmax(box):
    xs,ys = [],[]
    for i in range(len(box)):   
        xs.append(box[i][0])
        ys.append(box[i][1])
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys) 
    return (min_x,min_y),(max_x,max_y)

'''check if the box drawn has flakes or not'''
    
def if_in_flake(box, refPt):  
    #for the refPt
    l1 = minmax(refPt)[0]
    r1 = minmax(refPt)[1]
    #for the flake's box
    l2 = minmax(box)[0]
    r2 = minmax(box)[1]
    if (l1[1] == r1[1] or l1[0] == r1[0] or l2[1] == r2[1] or l2[0] == r2[0]):		# the line cannot have positive overlap
        return False
    if (l1[1] >= r2[1] or l2[1] >= r1[1]):   # If one rectangle is on left side of other
        return False
    if (r1[0] <= l2[0] or r2[0] <= l1[0]):   # If one rectangle is above other
        return False
        
    return True  #if it's overlapping

''' get distance between 2 point''' ### use instead of math.dist function '''

def get_dist(box1, box2):
    x1, y1 = box1
    x2, y2 = box2
    return math.hypot(x1-x2,y1-y2)

#-----------------------------------------------------------------------------------------------------------------------------------------------------#
''' main detection code '''

def onephoto_detection(file_name, thres_min, thres_max):
    
    all_flake_area = []
    global data
    data = []
    file_path = find_file_path(file_name)[0]
    file_name_short = find_file_path(file_name)[1]
    scale = find_file_path(file_name)[2]
    ori_pic = cv.imread(file_path, )
    pic = ori_pic[0:960, 0:1280]  ### crop upper and lower part of image) ###
    b, g, r = cv.split(pic)   ### switch it to r, g, b ###
    pic = cv.merge([r, g, b])
    pic_gray = cv.cvtColor(pic, cv.COLOR_BGR2GRAY) 

    print("file_name:",file_path)
    
    ''' selecthreshold '''
    #-----------------------------------IMPORTANT-----------------------------------#
    ret, thres = cv.threshold(pic_gray,thres_min,thres_max,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
    (counts, _) = cv.findContours(thres, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)        
    #-----------------------------------IMPORTANT-----------------------------------#
    
    '''!!! select the size range of flakes to be detected!!!'''
    
    size_range = [0.1,200]
    #[2],30]       #!!!<- size selection !!!# already converted to micrometer; gotta be > 0.06 um (AAO)
    
    selected_counts = []
    for j in range(1,len(counts)):
        if len(counts)==1:
            break
        elif size_range[0] <= float(format(cv.contourArea(counts[j])/(scale**2),".4f")) <= size_range[1]:
            selected_counts.append(counts[j])
    
    ''' draw boundary line of selected count '''
    
    flakes = thres.copy()
    flakes_drawn = gray_to_RGB(thres.copy())
    cv.drawContours(flakes_drawn, selected_counts, -1, (106, 81, 255), 2)
   
    # plt.imshow(flakes_drawn), plt.title("flakes_counts: "+str(len(selected_counts)))
    # plt.show()
    
    index = 0
    for j in range(len(selected_counts)):
        x,y,w,h = cv.boundingRect(selected_counts[j])       ### create rectangle along the image ###
        flakes_gray = flakes.copy()
        flakes_gray = cv.merge([flakes_gray, flakes_gray,flakes_gray])
        rect = cv.minAreaRect(selected_counts[j])
        box = cv.boxPoints(rect)
        box = np.int0(box)
        slice_data = [index]
        slice_data.append(file_name_short)
        
        #### box_positions ###
        slice_data.append(box)
        
        ### area_each_flake counted by pixel ###
        slice_data.append(cv.contourArea(selected_counts[j])/(scale**2))   
        
        ### area_each_cal counted using wxh of box ###
        #slice_data.append(math.dist(box[0],box[1])*math.dist(box[1],box[2])/(scale**2))     ### function math.dis does not work (for now)
        slice_data.append(get_dist(box[0], box[1])*get_dist(box[1], box[2])/(scale**2))
        
        ### side_lengths
        #slice_data.append(( round(math.dist(box[0],box[1])/scale,3)  ,  round(math.dist(box[1],box[2])/scale,3) )   ### function math.dis does not work (for now)     )
        slice_data.append(round(get_dist(box[0],box[1])/scale,3) )
        
        slice_data.append(round(get_dist(box[1],box[2])/scale,3) ) 
                          
                          
        mask = np.zeros(pic.shape,np.uint8)
        cv.drawContours(mask,[selected_counts[j]],0,(106,81,255))
        pixelpoints = np.transpose(np.nonzero(mask))
        
        for i in range(4):
            slice_data.append(colors(pic,pixelpoints)[i])
            
        #### selected_counts from drawing then drop it eventually ###
        slice_data.append(selected_counts[j])

        text = "Area:" + format(cv.contourArea(selected_counts[j])/(scale**2),".4f")
        index += 1
        data.append(slice_data)
    
    if len(selected_counts) == 0:
        print("there's no flake T.T")
        return None
    
    ''' data collection in list '''
   
    data_keys = ["old_index","file_name","box_positions", "area_each_flake", "area_each_cal", "width", "length", "mean_color", "min_color", "max_color", "SD_color", "selected_counts"] ###--> when area each cal and side length will work ###
   
   #data_keys = ["old_index","file_name","box_positions", "area_each_flake", "mean_color", "min_color", "max_color", "SD_color", "selected_counts"]
    df = pd.DataFrame(data)
    pd.set_option('display.max_colwidth', 0)
    df.columns = data_keys
    df.index = [i for i in range(len(selected_counts))]
    
    ''' mouse selection '''
    
    def click_crop(event, x, y, flags, param):
        # grab references to the global variables
        global refPt, cropping
        if event == cv.EVENT_LBUTTONDOWN: ### mouse button clicked, recording started ###
            refPt = [(x, y)]
            cropping = True
        elif event == cv.EVENT_LBUTTONUP:  ### mouse released -> the cropping operation is finished ###
            refPt.append((x, y))
            cv.rectangle(image, refPt[0], refPt[1], (0, 255, 0), 2)  #<- draw ROI
            cv.imshow("image", image)
            roi = clone[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
            a.append(refPt)
            cropping = False

    refPt = []
    cropping = False
    a = [] 
    image = flakes_drawn
    clone = image.copy()
    cv.namedWindow("image")
    cv.setMouseCallback("image", click_crop)
    global new_selected_counts, data2
    data2 = data
    cnt = 0
    j=0
    while True:
        cv.imshow("image", np.concatenate((image, pic), axis=1))
        key = cv.waitKey(1) & 0xFF
        #print(df.drop(columns="selected_counts"))
        boxes = list(df["box_positions"]) 
        new_selected_counts = []
        square_new_flakes_drawn = gray_to_RGB(thres.copy())
        new_flakes_drawn = gray_to_RGB(thres.copy())
        
        while j < len(a):
            while i < len(list(df["box_positions"])):    ### boxes is list of all flakes' box_positions ###
                boxes = list(df["box_positions"]) 
                if ( if_in_flake( boxes[i] , a[j] ) ) == True: ### and df["old_index"][i]==i: #the cropping overlaps the flake boundary, delete it                    
                    df.drop(i, axis=0, inplace=True)
                    df.set_axis([i for i in range(df.shape[0])], axis='index', inplace=True)
                    break
                i+=1
            if i == len(list(df["box_positions"])):
                j+=1  

        new_selected_counts = list(df["selected_counts"])
        cv.drawContours(new_flakes_drawn, new_selected_counts, -1, (106, 81, 255), 3)
        for i in range(len(a)):
            cv.rectangle(square_new_flakes_drawn, a[i][0], a[i][1], (106, 81, 255), 2) 
            
        if key == ord("r"):          ### press r to reset ###
            image = clone.copy()
            data_keys = ["old_index","file_name","box_positions", "area_each_flake", "area_each_cal", "width", "length", "mean_color", "min_color", "max_color", "SD_color", "selected_counts"]
            #data_keys = ["old_index","file_name","box_positions", "area_each_flake", "mean_color", "min_color", "max_color", "SD_color", "selected_counts"]
            df = pd.DataFrame(data)
            pd.set_option('display.max_colwidth', 0)
            df.columns = data_keys
            df.index = [i for i in range(len(selected_counts))]

            a = []
            

        elif key == ord("c"):          ### press c to crop more ###
            image = new_flakes_drawn.copy()
            
        elif key == ord("f"):     ### press f to finish ###
            break

        elif key == ord("d"):      ### press d to delete --> cancle all ### !!! Change directory!!!
            image = gray_to_RGB(thres.copy()).copy()
            new_flakes_drawn = gray_to_RGB(thres.copy()).copy()
            cv.waitKey(1)
            cv.destroyAllWindows()  ### close all open windows ###
            cv.waitKey(1)
            plt.imshow(new_flakes_drawn)
            image_name = 'K:/Kanit/Postdoc at VISTEC/Project_size selection of MXene/Results/Python code/cascade 20220305/10000rpm/Not processed/' + file_name_short + '_.tiff'
            cv.imwrite(image_name, ori_pic)
            return None
        
    cv.waitKey(1)
    cv.destroyAllWindows()  # close all open windows
    cv.waitKey(1)

    ###------------------------------DONE FOR THE CROPPING-------------------------------###
    
    cv.namedWindow("result")
    cv.setMouseCallback("result", click_crop)
    while True:
        key = cv.waitKey(1) & 0xFF
        cv.imshow(file_name_short+"  new_selected_counts: "+str(len(new_selected_counts)), np.concatenate((new_flakes_drawn, pic), axis=1))
        if key == ord("f"):  
            break
    cv.waitKey(1)
    cv.destroyAllWindows()  # close all open windows
    cv.waitKey(1)
    plt.imshow(new_flakes_drawn), plt.title("Total Flake Count is: "+str(len(new_selected_counts)))
    plt.show()
    
    ###################d#problem######################
    
    image_name = 'K:/Kanit/Postdoc at VISTEC/Project_size selection of MXene/Results/Python code/cascade 20220305/10000rpm/Processed/' + file_name_short + '.tiff'
    cv.imwrite(image_name, new_flakes_drawn)
     
    return df.drop(columns=["selected_counts","mean_color", "max_color","min_color","SD_color"]).set_axis([i for i in range(len(df.index))], axis='index')
    
x = onephoto_detection(file_name="6000rpm_sonic-1-1k.tif",  thres_min = - 0, thres_max = 205)


#%% separate cell

''' analyze multiple images in one folder '''

import numpy as np
num_bins = 20

#for i in range(1): #for i in range(len(test_folders)) ### i = number of folders in the directory ###
#for i in range(1,2):
    #real_test_list = os.listdir(test_directory+"/"+test_folders[i])

real_test_list = os.listdir(test_directory+"/10000rpm")
real_test_list.sort()
result_dataframe = pd.DataFrame()
j = 0
for j in range(16,18):     ### range(1,len(real_test_list)): ### j = number of files in the folder ### change number in (start,end) for files in folder 
        x = onephoto_detection(file_name = real_test_list[j],  thres_min = 0, thres_max =255)
        if x is None:
            j+=1
        else:
            result_dataframe = pd.concat([result_dataframe,x])
   
    #print(result_dataframe)
n, bisns, patches = plt.hist(list(result_dataframe["area_each_flake"]), num_bins, range=(2,50), facecolor='blue', alpha=0.5)
plt.xlabel("Area(um^2)", fontsize=12), plt.ylabel("Count", fontsize=12), plt.title("Total flake numbers: " + str(len(result_dataframe.index)))


plt.show()

### save data to excel --> need to change excel name if one folder needs many runs ###
result_dataframe.to_excel("K:/Kanit/Postdoc at VISTEC/Project_size selection of MXene/Results/Python code/cascade 20220305/10000rpm/10000rpm_table_4.xlsx", index = False)

 
result_dataframe

# %%
num_bins = 100
n, bins, patches = plt.hist(list(result_dataframe["area_each_flake"]), num_bins, range=(2,50), facecolor='blue', alpha=0.5)
plt.xlabel("Area(um^2)", fontsize=12), plt.ylabel("Count", fontsize=12), plt.title("Total flake numbers: " + str(len(result_dataframe.index)))
plt.show()

#%%
