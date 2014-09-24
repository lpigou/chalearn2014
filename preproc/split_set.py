from ChalearnLAPSample import GestureSample
from cPickle import dump
from glob import glob
from random import shuffle
import cv2
import os
import sys
import shutil
import errno
import gzip
import time
from itertools import tee, islice
from numpy import *
from numpy import linalg
from numpy.random import RandomState



data = "/home/lio/mp/chalearn2014/train_raw"
output = "/home/lio/Dropbox/MP/chalearn2014/preproc"
# data = "/home/lio/mp/chalearn2014/train_raw"

def write(_s): 
    with open(output+"/sample.txt","a") as f: f.write(_s+"\n")
    print _s

# get samples
os.chdir(data)
samples=glob("*.zip")
samples.sort()

for file in samples:
    print file,
    smp = GestureSample(data+"/"+file)
    # gestures = smp.getGestures()

    n = smp.data['numFrames']
    vid = smp.rgb

    for i in range(n):
    	img = vid.read()[1]
    	ratio = 1.*img.shape[1]/img.shape[0]
    	size=200 if img.shape[0]<200 else 400
    	img = cv2.resize(img, (int(size*ratio), size))
    	cv2.imshow("Video", img)
    	key =  cv2.waitKey(0)
    	if key==65505: 
    		break
    	elif key==13: 
    		write(file)
    		print "added"
    		break
    time.sleep(0.1)
