#!/usr/bin/env python3
#
# Author: Andrew Maule
# Objective: Test script for segmentation funcitonality.

import argparse
import numpy as np
import pandas as pd
import sys

#Local libs

#
# construct the argument parser and parse the arguments
def parse_args():
    parser = argparse.ArgumentParser(prog=sys.argv[0], description="Compute the difference b/w Matlab GiNA output and that of GiNADeux")
    parser.add_argument('--i1', '--input1', dest='input1', required=True, help="Path to Matlab GiNA output table.")
    parser.add_argument('--i2', '--input2', dest='input2', required=True, help="Path to GiNADeux output table.")
    parser.add_argument('-o', '--output', dest='output', required=True, help="Path to comparison table b/w two GiNA versions (CSV format)")
    parsed = parser.parse_args(sys.argv[1:])
    return(parsed)

if __name__ == '__main__':
    parsed = parse_args();
    ginaUnTable     = pd.read_csv(parsed.input1)
    ginaUnTable     = ginaUnTable.set_index(['picture', 'numbering'])
    ginaDeuxTable   = pd.read_csv(parsed.input2)
    ginaDeuxTable   = ginaDeuxTable.set_index(['picture', 'numbering'])
    commonFields    = np.intersect1d(ginaUnTable.columns,ginaDeuxTable.columns)
    diffTable       = ginaUnTable[commonFields] - ginaDeuxTable[commonFields]
    diffTable.to_csv(parsed.output)
