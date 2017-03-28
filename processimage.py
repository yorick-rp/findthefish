# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 12:28:13 2017

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


def sobelFilter(img):
  imgsobel = filters.sobel(img)
  return imgsobel
#  imgray = img # maybe convert to grayscale first?
#  imx = zeros(imgray.shape)
#  filters.sobel(imgray,1,imx)
#  imy = zeros(imgray.shape)
#  filters.sobel(imgray,0,imy)
#  magnitude = sqrt(imx**2+imy**2)
#  return magnitude.astype(int)
# enddef


def sobelCorners(img):

  imgray = img # maybe convert to grayscale first?
  imx = zeros(imgray.shape)
  filters.sobel(imgray,1,imx)
  imy = zeros(imgray.shape)
  filters.sobel(imgray,0,imy)
  prodsobel = np.multiply(imx,imy)
  prodsobel = np.round((prodsobel-np.min(prodsobel))/np.max(prodsobel)*255)
  return prodsobel
# enddef


def gaussianBlur(img,sigma=3):
  imgfiltered = filters.gaussian_filter(img, sigma)
  return imgfiltered
# enddef


def gaussDer(img,sigma=3,k=sqrt(2)):
  img_gsigma = filters.gaussian_filter(img,sigma)
  img_gksigma = filters.gaussian_filter(img,k*sigma)
  #img_gk2sigma = filters.gaussian_filter(imgray,k*k*sigma)
  return img_gksigma-img_gsigma
# enddef
 
# 2 level segmentation, find ideal threshold
def binarySegmentation(img):
  imgflat = img.flatten()
  T = 100 
  lastT = 0
  while (abs(T - lastT) > 1): 
      lastT = T
      mf = mean(imgflat[np.where(imgflat>T)])
      mb = mean(imgflat[np.where(imgflat<=T)])
      T = 0.5*(mb+mf)
  #endwhile
  imgsegm = 1*(img<T)
  return imgsegm
# enddef

#grayscale
def kMeansSegmentationGray(img,k=5):
  rows,cols = shape(img)

  R = img.reshape(cols*rows,1)
  
  kmeans = KMeans(n_clusters=k, random_state=0).fit(R)
  
  imsegmented = kmeans.labels_.reshape(rows,cols)
  #add colours:
  labels = kmeans.labels_
  levels = kmeans.cluster_centers_
  imsegmented = (levels[np.array(labels)]).reshape(rows,cols).astype(uint8)
  return imsegmented
# enddef

# color
def kMeansSegmentation(img,k=5):
  rows,cols, planes = shape(img)

  R = img.reshape(cols*rows,planes)
  
  kmeans = KMeans(n_clusters=k, random_state=0).fit(R)
  
  imsegmented = kmeans.labels_.reshape(rows,cols)
  #add colours:
  labels = kmeans.labels_
  levels = kmeans.cluster_centers_
  imsegmented = (levels[np.array(labels)]).reshape(rows,cols,planes).astype(uint8)
  return imsegmented
# enddef

# denoise: in imtools
def denoise(imgray):
  return imtools.denoise(imgray,imgray)
# enddef


