# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 10:06:53 2017

@author: Yorick.Boheemen
"""

import os 
from scipy import ndimage
from subprocess import check_output

import cv2
import numpy as np
from matplotlib import pyplot as plt
#%matplotlib inline

from os import listdir
from os.path import isfile, join

import processimage



def matchTemplateAllMethods(img,template):
  img2 = img.copy()
  w, h = template.shape[::-1]

  # All the 6 methods for comparison in a list
  methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
              'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
  
  for meth in methods:
       #img = img2
       method = eval(meth)
   
       # Apply template Matching
       res = cv2.matchTemplate(img,template,method)
       min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
   
       # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
       if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
           top_left = min_loc
       else:
           top_left = max_loc
       bottom_right = (top_left[0] + w, top_left[1] + h)
   
       cv2.rectangle(img2,top_left, bottom_right, 255, 2)
       
       if meth == methods[-1]:
         fig, ax = plt.subplots(figsize=(12, 7))
         plt.imshow(img2,cmap = 'gray')
         plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
         plt.suptitle(meth)
         plt.show()
       # endif
     
  #     fig, ax = plt.subplots(figsize=(12, 7))
  #     plt.subplot(121),plt.imshow(res,cmap = 'gray')
  #     plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
  #     plt.subplot(122),plt.imshow(img2,cmap = 'gray') #,aspect='auto'
  #     plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
  #     plt.suptitle(meth)
  # 
  #     plt.show()
  
  #endfor
  
# enddef

def process(img,filter='none'):
  if filter == 'none':
    return img
  elif filter == 'sobel':
    return processimage.sobelFilter(img).astype(int)
  elif filter == 'blur':
    return processimage.gaussianBlur(img)
  elif filter == 'gausian difference':
    return processimage.gaussDer(img)
  elif filter == 'binary segmentation':
    return processimage.binarySegmentation(img)
  elif filter == 'k-means segmentation':
    return processimage.kMeansSegmentationGray(img)
  elif filter == 'denoise':
    return processimage.denoise(img)
  
  raise Exception ('no valid filter chosen')
# enddef


#imgpath = r'C:\Users\yorick.boheemen\Documents\Kaggle\WhereIsTheFish' 
imgfolder = r'O:\DataTeam\FindTheFish\Data\train\train\ALB\YVB'
imgfilenames = [f for f in listdir(imgfolder) if isfile(join(imgfolder, f))]
imgfilenames = imgfilenames[0:5]
#imgname = imgpath + '\\' + 'img_01512.jpg'
imgname = imgfolder + '\\' + 'img_00003.jpg' # imgfilenames[0]

#imgpathcrop = r'C:\Users\yorick.boheemen\Documents\Kaggle\WhereIsTheFish' 
imgfoldercrop = r'O:\DataTeam\FindTheFish\Data\train\train\ALB\YVB\crop'
imgcropfilenames = [f for f in listdir(imgfoldercrop) if isfile(join(imgfoldercrop, f))]
#imgcropname = imgpathcrop + '\\' + 'img_00091_crop.jpg'
templatefilenames = ['img_00003.jpg'   # original fish
                     ,'img_00043.jpg'  # vertical
                     ,'img_00055.jpg'  # sun hat
                     ,'img_00163.jpg'  # leg before stomach
                     ,'img_00191.jpg' # diagonal
                     ,'img_00227.jpg'
                     ,'img_00299.jpg'
                     ,'img_00215.jpg'
                     ,'img_00039.jpg'
                     ]
imgcropname = imgfoldercrop + '\\' +templatefilenames[2]
# imgcropfilenames[0]

# load image (and crop):
#im_array = cv2.imread(imgcropname,0)
#img_rows,img_cols = im_array.shape
#template = np.zeros([ img_rows, img_cols], dtype='uint8') # initialisation of the template
#
## skip next step because image is already cropped                   
#template[:, :] = im_array#[100:450,525:950] # I try manually to find the correct rectangle. 
##template /= 255.
#plt.subplots(figsize=(10, 7))
#plt.subplot(121),plt.imshow(template, cmap='gray') 
#plt.subplot(122), plt.imshow(im_array, cmap='gray')

processingFilter = 'sobel'

for templatefilename in templatefilenames:
  imgtemplatepath = imgfoldercrop + '\\' +templatefilename
  imgtemplate = cv2.imread(imgtemplatepath,0)
  imgtemplate = process(imgtemplate,processingFilter)
  plt.subplots(figsize=(8, 5))
  plt.imshow(imgtemplate, cmap='gray') 
  plt.show()
  
  for imgfilename in imgfilenames:
    imgpath = imgfolder + '\\' +imgfilename
    img = cv2.imread(imgpath,0)
    img = process(img,processingFilter)
    matchTemplateAllMethods(img,imgtemplate)
  # endfor
# endfor
    
  

imgtemplatepath = imgfoldercrop + '\\' +templatefilenames[0]
imgtemplate = cv2.imread(imgtemplatepath,0)
imgsobel = processimage.sobelFilter(imgtemplate)#.astype(int)


plt.imshow(imgtemplate)
plt.imshow(processimage.sobelFilter(imgtemplate), cmap='gray')
plt.imshow(processimage.gaussianBlur(imgtemplate), cmap='gray')
plt.imshow(processimage.gaussDer(imgtemplate), cmap='gray')
plt.imshow(processimage.binarySegmentation(imgtemplate), cmap='gray')
plt.imshow(processimage.kMeansSegmentationGray(imgtemplate), cmap='gray')
imgdenoised,bla = processimage.denoise(imgtemplate)
plt.imshow(imgdenoised)
imgtemplate.shape



