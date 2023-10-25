# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
import os
from datasets import load_dataset




def load_data(path):
    dataset = load_dataset("csv", data_files = {"train": path})
    #dataset = dataset.shuffle()

    return dataset


    