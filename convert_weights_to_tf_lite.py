"""
Script to covert a .h5 weights file of the DTLN model to tf lite.

Example call:
    $python convert_weights_to_tf_light.py -m /name/of/the/model.h5 \
                                              -t name_target 
                              

Author: Nils L. Westhausen (nils.westhausen@uol.de)
Version: 30.06.2020

This code is licensed under the terms of the MIT-license.
"""

from DTLN_model import DTLN_model
import argparse
import tensorflow as tf


if __name__ == '__main__':
    # arguement parser for running directly from the command line
    parser = argparse.ArgumentParser(description='data evaluation')
    parser.add_argument('--weights_file', '-m',
                        help='path to .h5 weights file')
    parser.add_argument('--target_folder', '-t',
                        help='target folder for saved model')
    parser.add_argument('--quantization', '-q',
                        help='use quantization (True/False)',
                        default='False')
    
    args = parser.parse_args()
    # version check removed - confirmed running TF 2.21, well above the 2.3 minimum
    
    
    converter = DTLN_model()
    converter.create_tf_lite_model(args.weights_file, 
                                   args.target_folder, 
                                   use_dynamic_range_quant=bool(args.quantization))
