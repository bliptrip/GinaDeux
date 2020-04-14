#!/usr/bin/env python3
#
# Author: Andrew Maule
# Objective: Abstract base classes for image segmentation.
#
from abc import ABC, abstractmethod
import cv2
#import imutils
import numpy as np
from skimage.filters.rank import mean
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, disk, flood_fill

class Segment():
    def __init__(self, resize=0.5, minArea=1500, maxArea=3000):
        super().__init__()
        self.resize = resize
        self.minArea = minArea
        self.maxArea = maxArea
        return

    def preprocess(self, image):
        image = cv2.resize(image, (0,0), fx=self.resize, fy=self.resize) 
        avg = np.zeros(image.shape, dtype='uint8')
        for c in range(image.shape[2]):
            avg[:,:,c] = mean(image[:,:,c], disk(4))
        return(avg)

    def postprocess(self, binimage):
        numOnes = np.where(binimage)[0].size
        numZeros = binimage.size - numOnes
        if( numOnes > numZeros ):
            binimage = ~binimage #Invert -- Why, I'm not sure, but this was in original GiNA code -- maybe had to do with putting green berries on black background
        binimage = closing(binimage, disk(3)) #Luis did this image postprocessing to remove noise
        #bwareaopen doesn't have a direct implementation.  Could just fill in 'holes', and then remove anything that
        #is below a given size in the segmention stage
        holes = binimage.copy().astype('uint8')*255
        holes = flood_fill(holes, (0,0), 255)
        # invert holes mask, img fill in holes
        holes = ~holes
        binimage = binimage | holes.astype('bool')
        return(clear_border(binimage))

    def segment(self, binimage):
        '''
        Returns a set of region properties on labeled image using skimage
        '''
        label_image = label(binimage)
        #Extract regions/blobs
        regions = filter(lambda prop: (prop.area >= self.minArea) and (prop.area <= self.maxArea), regionprops(label_image))
        #Extract contours/boundaries of each blob
        return(regions)
