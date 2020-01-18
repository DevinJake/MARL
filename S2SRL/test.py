# !/usr/bin/env python3
# The file is used to predict the action sequences for full-data test dataset.
import argparse
import logging
import sys
import random
from libbots import data, model, utils

import torch
log = logging.getLogger("data_test")

DIC_PATH = '../data/auto_QA_data/share.question'

if __name__ == "__main__":
    for epoch in range(3):
        list = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
        random.shuffle(list)
        for batch in data.iterate_batches(list, 5):
            print(batch)
        print(str(epoch)+'------------------------------------')
        for batch in data.iterate_batches(list, 5):
            print(batch)
        print(str(epoch)+'------------------------------------')