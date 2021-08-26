# -*- coding: utf-8 -*-
import argparse
import os

parse = argparse.ArgumentParser()
parse.add_argument('-e', '--epoch', help='epoch for iter', type=int)
args = parse.parse_args()
print(args.epoch ** 2)