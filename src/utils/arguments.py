'''
Arguments used in various scripts
'''
import argparse

def input_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--recording', '-r', type=int, default=0, help="Number of the recording to be used")
    args = parser.parse_args()
    return args