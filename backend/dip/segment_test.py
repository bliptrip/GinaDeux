#!/usr/bin/env python3
#
# Author: Andrew Maule
# Objective: Test script for segmentation funcitonality.

import argparse
import cv2
import sys

#Local libs
from binary_segment import BinaryThresholdSegment
from nn_segment import NNSegment

#
# construct the argument parser and parse the arguments
def parse_args():
    parser = argparse.ArgumentParser(prog=sys.argv[0], description="A command-line utility for executing unit testing on GiNA image segmentation implementations.")
    parser.add_argument('-i', '--input', '--input_image', dest='input', required=True, help="Path to input image.")
    parser.add_argument('-o', '--output', '--output_image', dest='output', required=True, help="Path to output (segmented) image.")
    parser.add_argument('--resize', type=float, default='0.5', help="Resize factor on original image before doing image processing.")
    parser.add_argument('--min_area', '--mina', '--minArea', dest='mina', type=int, default='1500', help="Remove foreground blobs (fruit) less than this size.")
    parser.add_argument('--max_area', '--maxa', '--maxArea', dest='maxa', type=int, default='1500', help="Remove foreground blobs (fruit) greather than this size.")
    parser.add_argument('--algorithm', action='store', type=str, choices=['binary','neural'], help="Type of segmentation algorithm to use (binary thresholding, neural network, etc.")
    group = parser.add_argument_group('binary', 'Binary Threshold Parameters')
    group.add_argument('--threshold', type=int, default=100, help="Threshold level to segment foreground from background.")
    group.add_argument('--channel', type=int, default=3, help="Image channel to do the thresholding operation on.")
    group = parser.add_argument_group('neural', 'Use a foreground and background image to train a simple neural network for doing image segmentation.')
    group.add_argument('--neural_model', dest='model', type=str, default="model", help='If a model has been previously generated, a file path (without extension) to its JSON-representation + h5 weights.')
    group.add_argument('--foreground_image', dest='foreground', help='Path to foreground image.')
    group.add_argument('--background_image', dest='background', help='Path to background image.')
    parsed = parser.parse_args(sys.argv[1:])
    return(parsed)

if __name__ == '__main__':
    parsed = parse_args();
    input_image = cv2.imread(parsed.input)
    resizeFactor = parsed.resize
    mina = parsed.mina
    maxa = parsed.maxa
    if( parsed.algorithm == 'binary' ):
        threshold = parsed.threshold
        channel = parsed.channel
        segmenter = BinaryThresholdSegment( channel=channel, threshold=threshold, resize=resizeFactor, minArea=mina, maxArea=maxa )
    elif( parsed.algorithm == 'neural' ):
        if( parsed.foreground and parsed.background ): #Train
            segmenter = NNSegment( resize=resizeFactor, minArea=mina, maxArea=maxa )
            segmenter.train(parsed.foreground, parsed.background)
            segmenter.export(modelPath) #Write the trained model and its weights back to files
        else: #Load previously trained model
            segmenter = NNSegment( modelPath=parsed.model, resize=resizeFactor, minArea=mina, maxArea=maxa )
    binimage = segmenter.predict(input_image)
    greyimage = binimage.astype(np.uint8)
    greyimage[greyimage != 0] = 255
    cv2.imwrite(parsed.output, greyimage)
    regionprops = segmenter.segment(binimage)
    print(regionrpops)
