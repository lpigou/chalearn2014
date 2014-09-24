from numpy import *
from numpy.random import RandomState, randint, rand
from cPickle import load, dump
from gzip import GzipFile
from glob import glob
from time import time
import os, shutil, errno, string
from multiprocessing import Process, Semaphore, Lock, Value, Event


import matplotlib.pyplot as plt
enum = enumerate
p_pool = Semaphore(3)

def dump_batch(video,traject,ori,hei,labels,batch_idx,p_pool):
    #dump
    filename= "batch/valid/batch_100_"+str(batch_idx)+"_.zip"
    print filename
    file = GzipFile(filename, 'wb')
    dump((video,(traject,ori,hei),labels), file, -1)
    file.close()
    p_pool.release()
    

data = "/home/lio/mp/chalearn2014/32x128x128_train"

file_valid_samps = "samples_bg1.txt"
valid_samps = open(file_valid_samps,"rb")
valid_samps = (valid_samps.read()).split("\n")
for i,samp in enum(valid_samps): valid_samps[i] = samp.replace(".zip","_pre.zip")
os.chdir(data)
# files = glob("*.zip")
# files = list(set(files).difference(valid_samps))
files = valid_samps
print files
print len(files),"found"

start_time = time()

video = empty((100, 2, 3, 32, 128, 128),dtype="uint8")
traject = empty((100, 3, 32),dtype="uint8")
ori = empty((100, 4, 32),dtype="uint8")
hei = empty((100,),dtype="uint8")
labels = empty((100,),dtype="uint8")
pos = 0
batch_idx = 0
for path in files:
    file = GzipFile(path, 'rb')
    v,s,l = load(file)
    file.close()
    t,o,p = s

    le = len(l)
    if pos+le > 100: le = 100-pos

    print pos, le
    sli = slice(pos,pos+le)
    video[sli] = v[:le]
    traject[sli] = t[:le]
    ori[sli] = o[:le]
    hei[sli] = p[:le]
    labels[sli] = l[:le]
    pos += le
    if pos == 100:
        p_pool.acquire()
        Process(target=dump_batch, args=(video,traject,ori,hei,labels,batch_idx,p_pool)).start()
        video = empty((100, 2, 3, 32, 128, 128),dtype="uint8")
        traject = empty((100, 3, 32),dtype="uint8")
        ori = empty((100, 4, 32),dtype="uint8")
        hei = empty((100,),dtype="uint8")
        labels = empty((100,),dtype="uint8")
        pos = 0
        if len(l)>le:
            les = len(l)-le
            sli = slice(pos,pos+les)
            video[sli] = v[le:]
            traject[sli] = t[le:]
            ori[sli] = o[le:]
            hei[sli] = p[le:]
            labels[sli] = l[le:]
            print pos, les
            pos += les
        batch_idx += 1

if pos>0:
    filename= "batch/batch_"+str(pos)+"_"+str(batch_idx)+"_.zip"
    print filename
    file = GzipFile(filename, 'wb')
    dump((video[:pos],(traject[:pos],ori[:pos],hei[:pos]),labels[:pos]), file, -1)
    file.close()

print v.shape
print "total time:",time()-start_time

