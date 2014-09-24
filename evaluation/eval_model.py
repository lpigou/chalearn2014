import sys
sys.path.append('..')
from convnet3d import ConvLayer, NormLayer, PoolLayer, LogRegr, HiddenLayer, \
                    DropoutLayer, relu
# from data_aug import load_normal, start_load

# various imports
from cPickle import dump, load
from glob import glob
from time import time, localtime
from gzip import GzipFile
import os
import shutil
import string
import gzip
import cv2

# numpy imports
from numpy import ones, array, prod, zeros, empty, inf, float32, random, transpose

# theano imports
from theano import function, config, shared
from theano.tensor.nnet import conv2d
from theano.tensor import TensorType
import theano.tensor as T


def build(cpu):

    # constants
    floatX = config.floatX
    enum = enumerate

    file = gzip.GzipFile("params.zip", 'rb')
    params = load(file)
    file.close()
    # print params
    W = [[None,None],[None,None],[None,None]]
    b = [[None,None],[None,None],[None,None]]
    W[0][0],b[0][0],W[0][1],b[0][1],W[1][0],b[1][0],W[1][1],b[1][1],W[2][0],b[2][0],W[2][1],b[2][1],Wh,bh,Ws,bs = params

    #-----------------------------FLIP KERNEL------------------------------------------
    if cpu:
        W = array(W)
        W_new = [[None,None],[None,None],[None,None]]
        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                # w = W[i,j].get_value()
                w = W[i,j]
                print w.shape, w.dtype
                for k in range(w.shape[0]):
                    for l in range(w.shape[1]):
                        for m in range(w.shape[2]):
                            w[k,l,m] = cv2.flip(w[k,l,m],-1)
                W_new[i][j] = shared(array(w, dtype=floatX), borrow=True)
        W = W_new
        for i in range(len(b)):
            b_ = b[i]
            for j in range(len(b_)):
                b[i][j] = shared(array(b[i][j], dtype=floatX), borrow=True)
    #-----------------------------FLIP KERNEL------------------------------------------


    rng = random.RandomState(1337) # this will make sure results are always the same
    batch_size = 1

    in_shape = (1,2,2,32,64,64) # (batchsize, maps, frames, w, h) input video shapes 
    traj_shape = (batch_size,3,32) # (batchsize, input shape of the trajectory

    # hyper parameters
    # ------------------------------------------------------------------------------

    # use techniques/methods
    class use:
        drop = True # dropout
        depth = True # use depth map as input
        aug = False # data augmentation
        load = False # load params.p file
        traj = False # trajectory
        trajconv = False # convolutions on trajectory
        valid2 = False
        fast_conv = not cpu
        norm_div = True

        norm = True # normalization layer
        mom = True # momentum

    # regularization
    class reg:
        L1_traj = .0 # degree/amount of regularization
        L2_traj = .0 # 1: only L1, 0: only L2
        L1_vid = .0 # degree/amount of regularization
        L2_vid = .0 # 1: only L1, 0: only L2

    class trajconv:
        append = False # append convolutions result to original traject
        filter_size = 5
        layers = 3 # number of convolution layers
        res_shape = traj_shape[-1]-layers*(filter_size-1)

    class net:
        shared_stages = [] # stages where weights are shared
        shared_convnets = [] # convnets that share weights ith beighbouring convnet
        n_convnets = 2 # number of convolutional networks in the architecture
        maps = [2,16,32,48] # feature maps in each convolutional network
        # maps = [2,5,25,25] # feature maps in each convolutional network
        kernels = [(1,5,5), (1,5,5), (1,4,4)] # convolution kernel shapes
        pools = [(2,2,2), (2,2,2), (2,2,2)] # pool/subsampling shapes
        W_scale = [[0.04,0.04],[0.04,0.04],[0.04,0.04],0.01,0.01]
        b_scale = [[0.2,0.2],[0.2,0.2],[0.2,0.2],0.1,0.1]
        # scaler = [[33,24],[7.58,7.14],[5,5],1,1]
        scaler = [[1,1],[1,1],[1,1],1,1]
        stride = [1,1,1]
        hidden_traj = 64 # hidden units in MLP
        hidden_vid = 1024 # hidden units in MLP
        norm_method = "lcn" # normalisation method: lcn = local contrast normalisation
        pool_method = "max" # maxpool
        fusion = "early" # early or late fusion
        hidden = hidden_traj+hidden_vid if fusion=="late" else 512 # hidden units in MLP
        n_class = 21

    activation = relu
    n_stages = len(net.kernels)
    video_shapes = [in_shape[-3:]]

    def _shared(val, borrow=True):
        return shared(array(val, dtype=floatX), borrow=borrow)

    def ndtensor(n): return TensorType(floatX, (False,)*n) # n-dimensional tensor

    for i in xrange(n_stages):
        k,p,v = array(net.kernels[i]), array(net.pools[i]), array(video_shapes[i])
        conv_s = tuple(v-k+1)
        video_shapes.append(tuple((v-k+1)/p))
    n_in_MLP = net.maps[-1]*net.n_convnets*prod(video_shapes[-1]) 


    def conv_args(stage, i):
        """ ConvLayer arguments, i: stage index """
        args = {
            'batch_size':1, 
            'activation':activation, 
            'rng':rng,
            'n_in_maps':net.maps[stage],
            'n_out_maps':net.maps[stage+1], 
            'kernel_shape':net.kernels[stage], 
            'video_shape':video_shapes[stage],
            "fast_conv":use.fast_conv,
            "layer_name":"Conv"+str(stage)+str(i),
            "W_scale":net.W_scale[stage][i],
            "b_scale":net.b_scale[stage][i],
            "stride":net.stride[stage],
            "W":W[stage][i],
            "b":b[stage][i]
        }
        return args

    # print conv_args(0,0)
    x = ndtensor(len(in_shape))(name = 'x') # video input


    def var_norm(_x,imgs=True,axis=[-3,-2,-1]):
        if imgs:
            return (_x-T.mean(_x,axis=axis,keepdims=True))/T.maximum(1e-4,T.std(_x,axis=axis,keepdims=True))
        return (_x-T.mean(_x))/T.maximum(1e-4,T.std(_x))

    def std_norm(_x,axis=[-3,-2,-1]):
        return _x/T.maximum(1e-4,T.std(_x,axis=axis,keepdims=True))

    out = [x[:,0], x[:,1]]

    for stage in xrange(n_stages):
        for i in xrange(len(out)): # for each convnet of the stage
            if stage==0:
                gray_norm = NormLayer(out[i][:,0:1], method="lcn", kernel_size=9,
                    use_divisor=use.norm_div).output
                depth_norm = var_norm(out[i][:,1:2])
                out[i]  = T.concatenate([gray_norm,depth_norm],axis=1)
            elif stage==1:
                out[i] = NormLayer(out[i], method="lcn", kernel_size=7,
                    use_divisor=use.norm_div).output
            else:
                out[i] = var_norm(out[i],axis=[-3,-2,-1])
            out[i] = ConvLayer(out[i], **conv_args(stage, i)).output
            out[i] = PoolLayer(out[i], net.pools[stage], method=net.pool_method).output

    out = [out[i].flatten(2) for i in range(len(out))]
    out = T.concatenate(out, axis=1)
    out = var_norm(out,axis=1)


    #hidden layer
    out = HiddenLayer(out, 
            W = Wh,
            b = bh,
            n_in=n_in_MLP, 
            n_out=net.hidden, 
            rng=rng, 
            activation=activation,
            W_scale=net.W_scale[-2],
            b_scale=net.b_scale[-2]).output
    def lin(X): return X
    logreg = LogRegr(out, 
        W = Ws,
        b = bs,
        rng=rng, 
        activation=lin, 
        n_in=net.hidden, 
        n_out=net.n_class,
        W_scale=net.W_scale[-1],
        b_scale=net.b_scale[-1])

    pred = logreg.p_y_given_x

    x_ = _shared(empty(in_shape))

    print "compiling..."
    eval_model = function([], [pred], 
        givens={x:x_},
        on_unused_input='ignore')
    print "compiling done"

    return eval_model, x_

def build2():
    
    # constants
    floatX = config.floatX
    enum = enumerate

    file = gzip.GzipFile("params.zip", 'rb')
    params = load(file)
    file.close()
    print params
    W = [[None,None,],[None,None],[None,None]]
    b = [[None,None],[None,None],[None,None]]
    W[0][0],b[0][0],W[0][1],b[0][1],W[1][0],b[1][0],W[1][1],b[1][1],W[2][0],b[2][0],W[2][1],b[2][1],Wh,bh,Ws,bs = params

    rng = random.RandomState(1337) # this will make sure results are always the same
    batch_size = 1

    in_shape = (1,2,2,32,64,64) # (batchsize, maps, frames, w, h) input video shapes 
    traj_shape = (batch_size,3,32) # (batchsize, input shape of the trajectory

    # hyper parameters
    # ------------------------------------------------------------------------------

    # use techniques/methods
    class use:
        drop = True # dropout
        depth = True # use depth map as input
        aug = False # data augmentation
        load = False # load params.p file
        traj = False # trajectory
        trajconv = False # convolutions on trajectory
        valid2 = False
        fast_conv = True
        norm_div = False

        norm = True # normalization layer
        mom = True # momentum

    # regularization
    class reg:
        L1_traj = .0 # degree/amount of regularization
        L2_traj = .0 # 1: only L1, 0: only L2
        L1_vid = .0 # degree/amount of regularization
        L2_vid = .0 # 1: only L1, 0: only L2

    class trajconv:
        append = False # append convolutions result to original traject
        filter_size = 5
        layers = 3 # number of convolution layers
        res_shape = traj_shape[-1]-layers*(filter_size-1)

    class net:
        shared_stages = [] # stages where weights are shared
        shared_convnets = [] # convnets that share weights ith beighbouring convnet
        n_convnets = 2 # number of convolutional networks in the architecture
        maps = [2,16,32,64] # feature maps in each convolutional network
        # maps = [2,5,25,25] # feature maps in each convolutional network
        kernels = [(1,7,7), (1,8,8), (1,6,6)] # convolution kernel shapes
        pools = [(2,2,2), (2,2,2), (2,2,2)] # pool/subsampling shapes
        hidden_traj = 200 # hidden units in MLP
        hidden_vid = 300 # hidden units in MLP
        W_scale = 0.01
        b_scale = 0.1
        norm_method = "lcn" # normalisation method: lcn = local contrast normalisation
        pool_method = "max" # maxpool
        fusion = "early" # early or late fusion
        hidden = hidden_traj+hidden_vid if fusion=="late" else 500 # hidden units in MLP
        n_class = 21

    activation = relu
    n_stages = len(net.kernels)
    video_shapes = [in_shape[-3:]]

    def _shared(val, borrow=True):
        return shared(array(val, dtype=floatX), borrow=borrow)

    def ndtensor(n): return TensorType(floatX, (False,)*n) # n-dimensional tensor

    for i in xrange(n_stages):
        k,p,v = array(net.kernels[i]), array(net.pools[i]), array(video_shapes[i])
        conv_s = tuple(v-k+1)
        video_shapes.append(tuple((v-k+1)/p))
    n_in_MLP = net.maps[-1]*net.n_convnets*prod(video_shapes[-1]) 


    def conv_args(stage, i):
        """ ConvLayer arguments, i: stage index """
        args = {
            'batch_size':1, 
            'activation':activation, 
            'rng':rng,
            'n_in_maps':net.maps[stage],
            'n_out_maps':net.maps[stage+1], 
            'kernel_shape':net.kernels[stage], 
            'video_shape':video_shapes[stage],
            "fast_conv":use.fast_conv,
            "layer_name":"Conv"+str(stage),
            "W_scale":net.W_scale,
            "b_scale":net.b_scale,
            "stride":1,
            "W":W[stage][i],
            "b":b[stage][i]
        }
        return args

    # print conv_args(0,0)
    x = ndtensor(len(in_shape))(name = 'x') # video input


    def var_norm(_x,imgs=True,axis=[-3,-2,-1]):
        if imgs:
            return (_x-T.mean(_x,axis=axis,keepdims=True))/T.maximum(1e-4,T.std(_x,axis=axis,keepdims=True))
        return (_x-T.mean(_x))/T.maximum(1e-4,T.std(_x))
    def std_norm(_x,axis=[-3,-2,-1]):
        return _x/T.maximum(1e-4,T.std(_x,axis=axis,keepdims=True))

    out = [x[:,0], x[:,1]]
    norm_out = []
    conv_out = []
    pool_out = []

    for stage in xrange(n_stages):
        for i in xrange(len(out)): # for each convnet of the stage
            if stage==0: 
                gray_norm = NormLayer(out[i][:,0:1], method="lcn",use_divisor=False).output
                gray_norm = std_norm(gray_norm)
                depth_norm = var_norm(out[i][:,1:])
                out[i]  = T.concatenate([gray_norm,depth_norm],axis=1)
            else:
                out[i] = NormLayer(out[i], method="lcn",use_divisor=False).output
                out[i] = std_norm(out[i])

            norm_out.append(out[i].copy())

            out[i] = ConvLayer(out[i], **conv_args(stage, i)).output

            conv_out.append(out[i].copy())

            out[i] = PoolLayer(out[i], net.pools[stage], method=net.pool_method).output

            pool_out.append(out[i].copy())

    # out = [out[i].flatten(2) for i in range(len(out))]
    # out = T.concatenate(out, axis=1)

    # #hidden layer
    # out = HiddenLayer(out, 
    #         W = Wh,
    #         b = bh,
    #         n_in=n_in_MLP, 
    #         n_out=net.hidden, 
    #         rng=rng, 
    #         activation=activation).output

    # logreg = LogRegr(out, 
    #     W = Ws,
    #     b = bs,
    #     rng=rng, 
    #     activation=activation, 
    #     n_in=net.hidden, 
    #     n_out=net.n_class)

    # pred = logreg.p_y_given_x

    x_ = _shared(empty(in_shape))

    # eval_model = function([], [pred], 
    #     givens={x:x_},
    #     on_unused_input='ignore')

    visu_model = function([],norm_out+ conv_out+ pool_out, 
    # visu_model = function([],conv_out, 
        givens={x:x_},
        on_unused_input='ignore')
    return visu_model, x_