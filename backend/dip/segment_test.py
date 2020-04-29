#!/usr/bin/env python3
#
# Author: Andrew Maule
# Objective: Test script for segmentation funcitonality.

import argparse
import cv2
import numpy as np
import pandas as pd
from pathlib import Path,PurePath
import sys

#Local libs
from binary_segment import BinaryThresholdSegment
from nn_segment import NNSegment

#
# construct the argument parser and parse the arguments
def parse_args():
    parser = argparse.ArgumentParser(prog=sys.argv[0], description="A command-line utility for executing unit testing on GiNA image segmentation implementations.")
    group = parser.add_argument_group('inputs', 'Input Image Parameters')
    group.add_argument('-i', '--input', '--input_image', dest='input', help="Path to input image.")
    group.add_argument('--input_includes', '--includes', dest='includes', help="A newline-delimited list of images to process.")
    group.add_argument('--inputs_are_binary', action='store_true', help='If input image(s) are binary images, and thus do not explicitly do binary quantization, but instead calculate parameters on binary images.')
    parser.add_argument('-o', '--output', '--output_path', dest='output', default='./', help="A folder path to put the segmentation images, blobs, and table data into.  Defaults to the current working directory of the script.")
    parser.add_argument('-t', '--table', '--output_table', dest='table', help="Output image-derived metrics/traits to CSV file with --table <filename>.")
    parser.add_argument('--resize', type=float, default='0.5', help="Resize factor on original image before doing image processing.")
    parser.add_argument('--min_area', '--mina', '--minArea', dest='mina', type=int, default='2500', help="Remove foreground blobs (fruit) less than this size.")
    parser.add_argument('--max_area', '--maxa', '--maxArea', dest='maxa', type=int, default='30000', help="Remove foreground blobs (fruit) greather than this size.")
    parser.add_argument('--algorithm', action='store', type=str, choices=['binary','neural'], help="Type of segmentation algorithm to use (binary thresholding, neural network, etc.")
    group = parser.add_argument_group('binary', 'Binary Threshold Parameters')
    group.add_argument('--threshold', type=int, default=100, help="Threshold level to segment foreground from background.")
    group.add_argument('--channel', type=int, default=2, help="Image channel to do the thresholding operation on.")
    group = parser.add_argument_group('neural', 'Use a foreground and background image to train a simple neural network for doing image segmentation.')
    group.add_argument('--neural_model', dest='model', type=str, default="model", help='If a model has been previously generated, a file path (without extension) to its JSON-representation + h5 weights.')
    group.add_argument('--foreground_image', dest='foreground', help='Path to foreground image.')
    group.add_argument('--background_image', dest='background', help='Path to background image.')
    parsed = parser.parse_args(sys.argv[1:])
    return(parsed)

if __name__ == '__main__':
    parsed = parse_args();
    resizeFactor = parsed.resize
    mina = parsed.mina
    maxa = parsed.maxa
    if( parsed.algorithm == 'binary' ):
        threshold = parsed.threshold
        channel = parsed.channel
        segmenter = BinaryThresholdSegment( channel=channel, threshold=threshold, resize=resizeFactor, minArea=mina, maxArea=maxa, grid=True )
    elif( parsed.algorithm == 'neural' ):
        if( parsed.foreground and parsed.background ): #Train
            segmenter = NNSegment( resize=resizeFactor, minArea=mina, maxArea=maxa, grid=True )
            foreground = cv2.imread(parsed.foreground)
            background = cv2.imread(parsed.background)
            segmenter.train(foreground, background)
            segmenter.export(parsed.model) #Write the trained model and its weights back to files
        else: #Load previously trained model
            segmenter = NNSegment( modelPath=parsed.model, resize=resizeFactor, minArea=mina, maxArea=maxa, grid=True )

    assert (parsed.input or parsed.includes),"ERROR: Must specify either --input or --includes flag on command-line."
    if( parsed.input ):
        image_filenames = [parsed.input]
    elif( parsed.includes ):
        includes_fh = open(parsed.includes, 'r')
        image_filenames = [line.rstrip('\n') for line in includes_fh.readlines()]

    output_path = Path(parsed.output)
    if not output_path.exists():
        output_path.mkdir(parents=True)
    assert output_path.is_dir(), "Output path (--output_path) must be a directory."

    data_collated = pd.DataFrame();
    for image_filename in image_filenames:
        image_path = PurePath(image_filename)
        if not parsed.inputs_are_binary:
            input_image = cv2.imread(str(image_path))
            binimage = segmenter.predict(input_image)
            #NOTE: I originally output the binary image in greyscale JPEG-format, but when I reopened the image as a binary file subsequently, the JPEG algorithm + greyscale resulted in a different binary file being read in
            #Thus: Writing as lossless-compressed binary PNG files is the correct route
            cv2.imwrite(str(output_path / (image_path.stem + '.binary.png')), binimage.astype('uint8'), [cv2.IMWRITE_PNG_COMPRESSION, 5, cv2.IMWRITE_PNG_BILEVEL, 1, cv2.IMWRITE_PNG_STRATEGY, cv2.IMWRITE_PNG_STRATEGY_RLE]) #Output the segmented binary image, uncompressed bilevel PNG
        else:
            image_path_parent   = image_path.parent
            image_path_suffix   = image_path.suffix
            image_path_stem     = PurePath(image_path.stem).stem
            image_path_orig     = PurePath(image_path_parent) / (image_path_stem + '.JPG')
            input_image = cv2.imread(str(image_path_orig))
            binimage = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED).astype('bool')
        data, findex, rindex, contour_image, fruit_boundaries, reference_boundaries = segmenter.segment(input_image, binimage)
        cv2.imwrite(str(output_path / (image_path.stem + '.blobs' + image_path.suffix)), contour_image) #Output the segmented binary image
        if not parsed.inputs_are_binary:
            data['picture'] = pd.Series([image_path.name]*len(data), index=data.index)
        else: #Remove the 'binary' extension -- NOTE: This assumes the segmented binary file versions are labeled with '.binary.JPG' in suffix, or equivalent.
            image_path_suffix   = image_path.suffix
            image_path_stem     = PurePath(image_path.stem).stem
            data['picture']     = pd.Series([image_path_stem+image_path_suffix]*len(data), index=data.index)
        data['numbering']       = pd.Series([-1]*len(data.index), index=data.index)
        data.loc[findex, 'numbering'] = np.array(range(1,len(findex)+1),dtype='int')
        data_collated = data_collated.append(data.iloc[findex]) #Only append the detected fruits to the collated table.

    if( parsed.table ):
        data_collated.to_csv(str(output_path / parsed.table), index=False)
