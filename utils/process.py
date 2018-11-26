#!/usr/bin/env python
import os
import time
import numpy as np
import pandas as pd
from sys import stdout
from argparse import ArgumentParser
import json


class ProgressBar:
    '''
    Here I implement a progress bar program by myself.
    '''
    _bar = 50
    _blk = 0
    _n = 0
    def __init__(self):
        pass
        
    def setBar(self, num_iteration, bar_size = 50, bracket = '[]'):
        assert len(bracket) == 2
        self._blk = (num_iteration + bar_size - 1)/bar_size
        self._bar = num_iteration/self._blk
        self._n = num_iteration
        stdout.write("{1}{0}{2}".format(' '*self._bar, bracket[0], bracket[1]))
        stdout.flush()
        stdout.write("\b" * (self._bar + 1))
        
    def show(self, i):
        if((i+1) % self._blk == 0):
            stdout.write("=")
            stdout.flush()
        if(i+1 == self._n):
            stdout.write('\n')


def getClassWeights(y, n_class):
    t_start = time.time()
    max_cnt = 0
    clswts = {}
    length = len(y)
    ave = 1.*length/n_class
    print("\tStart computing class weights ...")
    pbar = ProgressBar()
    pbar.setBar(num_iteration=n_class)
    for i in range(n_class):
        pbar.show(i)
        cnt = list(y).count(i)
        max_cnt = max(max_cnt, cnt)
        clswts[i] = (ave/(cnt + 1.))**0.66
    print("\tTime usage for computing class_weights is: " + str(time.time()-t_start) + " sec")
    print("Max count is: " + str(max_cnt))
    return clswts


def rebuildCsv():
    parser = ArgumentParser()
    parser.add_argument(
            '--index', type=str, default="../input/train.csv",
            help='input index file')
    parser.add_argument(
            '--imgsrc', type=str, default="../data/train",
            help='the path of the image sources')
    parser.add_argument(
            '--outdir', type=str, default="../input",
            help='output dir where to put class_weights.json')
    o = parser.parse_args()
    t_start = time.time()
    assert os.path.exists(o.index), "The input index file does not exist!"
    assert os.path.exists(o.imgsrc), "The image src path does not exist!"
    print("Rebuilding index file ...")
    df = pd.read_csv(o.index)
    df.drop('url', axis=1, inplace=True)
    if "landmark_id" in df.columns:
        num_classes = df.landmark_id.max() + 1
        print("Total number of Class: " + str(num_classes))
        assert df.landmark_id.min() == 0, "What the fuck!"
        diff = set([i for i in range(num_classes)]) - set(df.landmark_id)
        print ("For the raw index file, number of missing classes: " + str(len(diff)))
    true_ids = [s.split('.')[0] for s in os.listdir(o.imgsrc)]
    df = df.loc[df.id.isin(true_ids)]
    if "landmark_id" in df.columns:
        diff = set([i for i in range(num_classes)]) - set(df.landmark_id)
        print ("For the rebuilt index file, number of missing classes: " + str(len(diff)))
        jfile = open(o.outdir + "/class_weights.json", 'w')
        json.dump(getClassWeights(df.landmark_id.tolist(), num_classes), jfile)
        jfile.close()
    outfile = o.index[:-4] + "_processed.csv"
    df.to_csv(outfile, index=False)
    print("Time Usage for rebuild index file is: " + str(time.time() - t_start) + " sec")


if __name__ == "__main__":
    rebuildCsv()
    