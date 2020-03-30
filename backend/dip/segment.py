#!/usr/bin/env python3
#
# Author: Andrew Maule
# Objective: Abstract base classes for image segmentation.
#
from abc import ABC, abstractmethod
import cv2
import imutils
import numpy as np
from skimage.filters.rank import mean
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, disk, flood_fill

class Segment(ABC):
    def __init__(self, resize=0.5, minArea=1500, maxArea=3000, minIntertia=0.2):
        super().__init__()
        self.resize = resize
        self.minArea = minArea
        self.maxArea = maxArea
        self.minIntertia = minInertia
        return

    def preprocess(self, image):
        image = cv2.resize(image, 0, self.resize, self.resize) 
        avg = mean(image, disk(4))
        return(avg)

    def postprocess(self, binimage):
        #Luis' original post-processing code
        #clear B
        #clear binaryImage
        #sOnes=length(find(mark6==1));
        #sZeros=length(find(mark6==0));
        #if sOnes>sZeros
        #    B=1+(mark6*-1);
        #    warNeg='col conv implemented)';
        #else
        #    B=mark6;
        #    warNeg='col conv was not requiered)';
        #end
        #se = strel('disk',3);
        #binaryImage = imclose(B,se);
        #binaryImage = imclearborder(bwareaopen(binaryImage,minApx));
        numOnes = np.where(binimage).size
        numZeros = binimage.size - numOnes
        if( numOnes > numZeros ):
            binimage = !binimage #Invert -- Why, I'm not sure, but this was in original GiNA code
        binimage = closing(binimage, disk(3,3)) #Luis did this image postprocessing to remove noise
        #bwareaopen doesn't have a direct implementation.  Could just fill in 'holes', and then remove anything that
        #is below a given size in the segmention stage
        holes = binimage.copy()
        flood_fill(holes, (0,0), True, inplace=True)
        # invert holes mask, img fill in holes
        holes = !holes
        binimage = binimage | holes
        return(clear_border(binimage))

    def segment(self, binimage):
        '''
        Returns a set of opencv contours using a binary contour image passed as input.
        '''
        label_image = label(binimage)
        regions = filter(lambda prop: (prop.area >= minArea) and (prop.area <= maxArea), regionprops(label_image))
        return(regions)
