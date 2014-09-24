import sys
# sys.path.append('..')
# from convnet3d import ConvLayer, NormLayer, PoolLayer, LogRegr, HiddenLayer, \
#                     DropoutLayer, relu
# from data_aug import load_normal, start_load

# various imports
from cPickle import dump, load
import gzip
import cv2
from glob import glob
from time import time, localtime
from gzip import GzipFile
import os
import shutil
import string, traceback

# numpy imports
from numpy import ones, array, prod, zeros, empty, inf, float32, random, transpose

# theano imports
from theano import function, config, shared
from theano.tensor.nnet import conv2d
from theano.tensor import TensorType
import theano.tensor as T

try:
	file = gzip.GzipFile("params_backup.zip", 'rb')
	params = load(file)
	file.close()

	floatX = config.floatX

	p = [shared(array(p_.get_value(), dtype=floatX), borrow=True) for p_ in params ]

	print p


	file = gzip.GzipFile("params.zip", 'wb')
	dump(p,file,-1)
	file.close()

except:
    print "".join(traceback.format_exception(*sys.exc_info()))
    raise Exception("".join(traceback.format_exception(*sys.exc_info())))