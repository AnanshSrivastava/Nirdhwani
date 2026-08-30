"""
Script to convert a .h5 weights file of the DTLN model to tf lite.

Example call:
    $python convert_weights_to_tf_lite.py -m /name/of/the/model.h5 \
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
    parser.add_argument('--weights_file', '-m', required=True,
                        help='path to .h5 weights file')
    parser.add_argument('--target_folder', '-t', required=True,
                        help='target folder for saved model')
    parser.add_argument('--quantization', '-q',
                        help='use quantization (True/False)',
                        default='False')
    
    args = parser.parse_args()
    # Version check removed; ensure TensorFlow >= 2.3 for TFLite conversion
    
    
    use_quant = str(args.quantization).strip().lower() in {"1", "true", "yes", "y"}
    converter = DTLN_model()
    converter.create_tf_lite_model(args.weights_file, 
                               args.target_folder,
                               use_dynamic_range_quant=use_quant)