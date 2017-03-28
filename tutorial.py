# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 10:33:55 2017

@author: Yorick.Boheemen
"""

from PIL import Image
from pylab import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import filters
from scipy.ndimage import measurements,morphology
from scipy import signal
import scipy.misc
from sklearn.cluster import KMeans
from os import listdir
from os.path import isfile, join

# custom imports:
import imtools
from imtools import denoise
from imtools import pca
#import cv2
import sift


imgpath = r'O:\DataTeam\FindTheFish\Data\train\train\ALB\YVB\crop'
imgfilenames = [f for f in listdir(imgpath) if isfile(join(imgpath, f))]

imgname = imgpath + '\\' + imgfilenames[0]
#img = r'..\img_00003_cropped.jpg'
# blur image:
imcolor = array(Image.open(imgname).convert('RGB'))
#imshow(imcolor)
imgray = array(Image.open(imgname).convert('L'))
figure()
gray()
#imshow(imgray)

#imblur = filters.gaussian_filter(imgray,5)
#imshow(imblur)

# blur color image:

#im2 = zeros(imcolor.shape)
#for i in range(3):
#    im2[:,:,i] = filters.gaussian_filter(imcolor[:,:,i],3)
#im2 = uint8(im2)
#imshow(im2)

# gradient filters:

#Sobel derivative filters
imx = zeros(imgray.shape)
filters.sobel(imgray,1,imx)
imy = zeros(imgray.shape)
filters.sobel(imgray,0,imy)
magnitude = sqrt(imx**2+imy**2)
imshow(magnitude)

#imsobel = zeros(imcolor[:,:,1].shape)
#for i in range(3):
#    imsinglecolor = imcolor[:,:,i]
#    imx = zeros(imsinglecolor.shape)
#    filters.sobel(imsinglecolor,1,imx)
#    imy = zeros(imsinglecolor.shape)
#    filters.sobel(imsinglecolor,0,imy)
#    magnitude = sqrt(imx**2+imy**2)
#    imshow(magnitude)
#    show()
#    imsobel += magnitude
##endfor
#imshow(imsobel)
    




# Gaussian derivatives:
    
sigma = 2 #standard deviation
imx = zeros(imgray.shape)
filters.gaussian_filter(imgray, (sigma,sigma), (0,1), imx)
imy = zeros(imgray.shape)
filters.gaussian_filter(imgray, (sigma,sigma), (1,0), imy)
magnitude = sqrt(imx**2+imy**2)
imshow(magnitude)


sigma=1.5
k=1.2
img_gsigma = filters.gaussian_filter(imgray,sigma)
img_gksigma = filters.gaussian_filter(imgray,k*sigma)
#img_gk2sigma = filters.gaussian_filter(imgray,k*k*sigma)
imshow(img_gksigma-img_gsigma)
#imshow(img_gk2sigma-img_gksigma)
#imshow(img_gk2sigma-img_gsigma)


# histogram:

hist(imgray.flatten(),128)
show()

# load image and threshold to make sure it is binary
#im = array(Image.open(’houses.png’).convert(’L’))

# thresholding/segmentation:

#imgrayflat = imgray.flatten()
#T = 100 
#lastT = 0
#while (abs(T - lastT) > 1): 
#    lastT = T
#    mf = mean(imgrayflat[np.where(imgrayflat>T)])
#    mb = mean(imgrayflat[np.where(imgrayflat<=T)])
#    T = 0.5*(mb+mf)
##endwhile
#
#imbound = 1*(imgray<T)
#imshow(imbound)
#labels, nbr_objects = measurements.label(imbound)
#print ("Number of objects:", nbr_objects)
#
#im_open = morphology.binary_opening(imgray,ones((9,5)),iterations=2)
#imshow(im_open)
#labels_open, nbr_objects_open = measurements.label(im_open)
#print ("Number of objects:", nbr_objects_open)


#k-means segmentation
rows,cols, planes = shape(imcolor)

R = imcolor.reshape(cols*rows,3)
k = 5

kmeans = KMeans(n_clusters=k, random_state=0).fit(R)

imsegmented = kmeans.labels_.reshape(rows,cols)
imshow(imsegmented)
#add colours:
labels = kmeans.labels_
levels = kmeans.cluster_centers_
imsegmented = (levels[np.array(labels)]).reshape(rows,cols,planes).astype(uint8)
imshow(imsegmented)


imsegmblur = filters.gaussian_filter(imsegmented,sigma=2)
imshow(imsegmblur)

np.unique(labels,return_counts=True)
#filter out level that occurs the most (in this case 0. is it always 0?)
#map the rest to 1
level = labels != 0
level = level.astype(int)
np.unique(level,return_counts=True)
levelimage = level.reshape(rows,cols)
imshow(levelimage)



testimgfile = r'O:\DataTeam\FindTheFish\Data\train\train\ALB\YVB\img_00003.jpg'
imgtest = array(Image.open(testimgfile).convert('L'))
figure()
gray()
imshow(imgtest)

imcorr = signal.correlate2d(imgtest,levelimage,boundary='wrap')
imshow(imcorr)
plot(imcorr[:,1400])

imcolor = imtools.denoise(imgray,imgray)


## SIFT

image_rgb = scipy.misc.imread(imgname)
sift_ocl = sift.SiftPlan(template=image_rgb, device=GPU)
kp = sift_ocl.keypoints(image_rgb)
kp.sort(order=["scale", "angle", "x", "y"])
print kp



img = cv2.imread(imgname)
gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
sift = cv2.xfeatures2d.SIFT_create()
kp = sift.detect(imgray,None)
imsift=cv2.drawKeypoints(imgray,kp,imgray)
cv2.imwrite('sift_keypoints.jpg',imsift)

imsiftkp=cv2.drawKeypoints(gray,kp,imsift,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imwrite('sift_keypoints.jpg',imsiftkp)





#denoise
U,imtextres = imtools.denoise(imgray,imgray)
imshow(U)
imshow(imtextres)


imshow(U*levelimage)

# pca

digitdata = pd.read_csv('digits_train.csv')
labels = digitdata['label']
digitdata = digitdata.iloc[:,1:]
len(digitdata.columns.values)
rows=28
cols=28
imshow(digitdata.iloc[3,:].reshape(cols,rows))

V = pca2(digitdata,7)
V.shape
V[1].shape
subplot(2,4,1)
imshow(immean.reshape(cols,rows))
for i in range(7):
    subplot(2,4,i+2)
    imshow(V[i].reshape(cols,rows))
show()

V,S,immean = pca(digitdata)
# show some images (mean and 7 first modes)
figure()
gray()
subplot(2,4,1)
imshow(immean.reshape(cols,rows))
for i in range(7):
    subplot(2,4,i+2)
    imshow(V[i].reshape(cols,rows))
show()
all(V.flatten())

