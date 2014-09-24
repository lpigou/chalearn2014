import sys
sys.path.append('../preproc')
from glob import glob
import gzip
from cPickle import load, dump
import os
import zipfile
import shutil
import cv2
from numpy import *
from ChalearnLAPSample import GestureSample
from scipy import misc
import preproc as pp
from scipy.signal import convolve2d
import eval_model

data = "/home/lio/mp/chalearn2014/train_raw"
hand = True
store = False
enum = enumerate

# import subprocess
# class VideoSink(object) :

#     def __init__( self, filename='output.avi', size=(512,512), rate=30, colorspace='rgb24',codec='lavc'):
 
#         # row/col --> x/y by swapping order
#         self.size = size[::-1]

#         cmdstring  = (  'mencoder',
#             '/dev/stdin',
#             '-demuxer', 'rawvideo',
#             '-rawvideo', 'w=%i:h=%i'%self.size+':fps=%i:format=%s'%(rate,colorspace),
#             '-o', filename,
#             '-ovc', codec,
#             '-nosound',
#             '-really-quiet'
#             )
#         self.p = subprocess.Popen(cmdstring, stdin=subprocess.PIPE, shell=False)

#     def __call__(self, frame) :
#         assert frame.shape[0:2][::-1] == self.size
#         # frame.tofile(self.p.stdin) # should be faster but it is indeed slower
#         self.p.stdin.write(frame.tostring())
#     def close(self) :
#         self.p.stdin.close()
#         self.p.terminate()

def main():
    params =load_gzip("params.zip")
    # print params
    if hand: 
        W = params[2].get_value()
        b = params[3].get_value()
    else: 
        W = params[0].get_value()
        b = params[1].get_value()
    # print W.shape


    # print params[3].get_value().shape
    # show_img(W[0,0,0],wait=0, norm=True)
    s = W.shape
    barv = zeros((s[-1],1),dtype="float32")
    group = 8
    W_o = W.copy()
    for k in xrange(W.shape[1]):
        for i in xrange(W.shape[0]/group):
            gr = barv
            for j in xrange(group):
                W[i*group+j,k,0] = cv2.flip(W[i*group+j,k,0],-1)
                img = W[i*group+j,k,0]
                cv2.normalize(img,img, 0, 255, cv2.NORM_MINMAX)
                # print gr.shape, img.shape
                gr = hstack([gr,img,barv])
                # print gr.shape
            barh = zeros((1,gr.shape[-1]),dtype="float32")
            if i==0 and k==0: res = zeros((3,gr.shape[-1]),dtype="float32")
            res = vstack([res,gr,barh])
        barh = zeros((2,gr.shape[-1]),dtype="float32")
        res = vstack([res,barh])

    # W = W.reshape((prod(s[:-1]),s[-1]))
    # print W.shape

    show_img(res, size=1000,wait=0)

    if store:
        visu_model, x_ = eval_model.build2()

        vid = process()
        print vid.shape
        x_.set_value(vid.astype("float32"), borrow=True)
        out = visu_model()
        file = gzip.GzipFile("convmaps.zip","wb")
        dump(out, file,-1)
        file.close()
    else:
        file = gzip.GzipFile("convmaps.zip","rb")
        out_ = load(file)
        file.close()
        for o in out_: print o.shape

        """
        0 normout L0
        (1, 2, 32, 64, 64) body
        (1, 2, 32, 64, 64) hand

        1 normout L1
        (1, 16, 16, 29, 29)
        (1, 16, 16, 29, 29)

        2 normout L2
        (1, 32, 8, 11, 11)
        (1, 32, 8, 11, 11)


        3 conv L0
        (1, 16, 32, 58, 58)
        (1, 16, 32, 58, 58)

        4 conv L1
        (1, 32, 16, 22, 22)
        (1, 32, 16, 22, 22)

        5 conv L2
        (1, 64, 8, 6, 6)
        (1, 64, 8, 6, 6)

        6 pool L0
        (1, 16, 16, 29, 29)
        (1, 16, 16, 29, 29)

        7 pool L1
        (1, 32, 8, 11, 11)
        (1, 32, 8, 11, 11)

        8 pool L2
        (1, 64, 4, 3, 3)
        (1, 64, 4, 3, 3)
        """
        vsk = VideoSink('/home/lio/Desktop/video2.avi',rate=20, 
            size=(480,640),colorspace='rgb24', codec="x264")

        for h_idx in [0,1]:
            out = array(out_)[[0*2+h_idx,3*2+h_idx,6*2+h_idx,1*2+h_idx,4*2+h_idx,7*2+h_idx,2*2+h_idx,5*2+h_idx,8*2+h_idx]]
            for k in range(9):
                # layer = k

                # idx = layer*2+h_idx

                vids = out[k][0]
                for fr in xrange(vids.shape[1]):
                    imgs = vids[:,fr]

                    # W = W_o
                    h = imgs.shape[-1]
                    barv = zeros((h,1),dtype="float32")+255
                    # group = imgs.shape[0]/2

                    # print k
                    if k in (0,):
                        group = 2.
                        wait=1
                    elif k in (1,2,3):
                        group = 4.
                        if k> 1: wait=2
                    elif k in (4,5,6):
                        group = 8.
                        if k> 4: wait=4
                    else:
                        group = 16.
                        if k> 7: wait=8

                    txt = ""
                    if k==0: txt = "Input ConvNet"
                    if k==1: txt = "Convoluties, laag 1"
                    if k==2: txt = "Max-pooling, laag 1"
                    if k==3: txt = "Normalisatie, laag 2"
                    if k==4: txt = "Convoluties, laag 2"
                    if k==5: txt = "Max-pooling, laag 2"
                    if k==6: txt = "Normalisatie, laag 3"
                    if k==7: txt = "Convoluties, laag 3"
                    if k==8: txt = "Max-pooling, laag 3"
                    # if k==0: txt = "Normalisatie, laag 2"


                    for i in xrange(int(ceil(imgs.shape[0]/group))):
                        gr = barv
                        for j in xrange(int(min(group,imgs.shape[0]))):
                            
                            img= imgs[i*group+j]
                            cv2.normalize(img,img, 0, 255, cv2.NORM_MINMAX)
                            img = img.astype("uint8")
                            # img = misc.imresize(img,(h2,h2))
                            gr = hstack([gr,img,barv])

                            # if fr==int(vids.shape[1]/2):
                            #     interp = cv2.INTER_NEAREST
                            #     img = cv2.resize(img.astype("uint8"), (256,256),interpolation=interp)
                            #     cv2.imwrite("imgs/im"+str(h_idx)+str(k)+str(i)+str(j)+".png", img,
                            #         (cv2.IMWRITE_PNG_COMPRESSION,0))

                            # print gr.shape
                        barh = zeros((1,gr.shape[-1]),dtype="float32")+255
                        if i==0: res = zeros((2,gr.shape[-1]),dtype="float32")+255
                        res = vstack([res,gr,barh])
                    barh = zeros((1,gr.shape[-1]),dtype="float32")+255
                    res = vstack([res,barh])
                    if res.shape[0]>res.shape[1]: 
                        size = 500
                    else: size = 1800
                    img = res


                    ratio = 1.*img.shape[0]/img.shape[1]

                    if ratio < 1:

                        shape = (640, int(ratio*640.))

                        img = cv2.resize(img, shape, interpolation=cv2.INTER_NEAREST)

                        a = zeros((480,640),dtype=uint8)
                        l = int(round(1.*(480.-shape[1])/2))
                        a[l:l+shape[1],:] = img
                        img=a
                    else:
                        shape = (int(ratio*480.), 480)


                        img = cv2.resize(img, shape, interpolation=cv2.INTER_NEAREST)
                        a = zeros((480,640),dtype=uint8)
                        l = int(round(1.*(640.-shape[0])/2))
                        a[:,l:l+shape[0]] = img
                        img=a

                    img = cv2.cvtColor(img,cv2.cv.CV_GRAY2RGB)
                    cv2.putText(img, txt, (10,470), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0),2)
                    # cv2.putText(img, txt, (10,470), cv2.FONT_HERSHEY_PLAIN, 1.5, (0,255,0),2)
                    #[, thickness[, lineType[, bottomLeftOrigin]]]) 
                    for i in range(wait):
                        # show_img(img, resize=False, wait=50)
                        vsk(img)



                    # if fr==int(vids.shape[1]/2):
                    #     interp = cv2.INTER_NEAREST
                    #     res = cv2.resize(res.astype("uint8"), 
                    #         (512,int(round(512.*float32(res.shape[0])/float32(res.shape[1])))),
                    #         interpolation=interp)

                    #     cv2.imwrite("imgs/+res"+str(h_idx)+str(k)+str(i)+str(j)+".png", res,
                    #         (cv2.IMWRITE_PNG_COMPRESSION,0))

def process():

    samples=glob(data+"/*.zip")
    # random.shuffle(samples)
    samples.sort()
    sample = samples[40]
    print sample
    sample = GestureSample(sample)
    gestures = sample.getGestures()
    gesture = gestures[3]

    skelet, depth, gray, user, c = pp.get_data(sample, gesture)
    user_o = user.copy()
    skelet,c = pp.proc_skelet(skelet)
    user = pp.proc_user(user)
    user_new, depth,c = pp.proc_depth(depth, user, user_o, skelet)
    gray,c = pp.proc_gray(gray, user,  skelet)
    user = user_new

    video = empty((1,3,)+gray.shape,dtype="uint8")
    video[0,0],video[0,1],video[0,2] = gray,depth,user

    v = array(video,dtype="uint8")
    v = v.swapaxes(1,2)
    # for i in xrange(gray.shape[1]):

    res_shape=(1,2,2,32,64,64)
    v_new = empty(res_shape,dtype="uint8")
    h = res_shape[-1]
    v = v[:,:,:res_shape[2]]

    p = skelet[3]
    if p < 10: p = 100
    ofs = p*0.25
    mid =  v.shape[-1]/2.
    sli = None
    if ofs < mid:
        start = int(round(mid-ofs))
        end = int(round(mid+ofs))
        sli = slice(start,end)

    for j in xrange(v.shape[2]): #maps
        for k in xrange(v.shape[3]): #frames
            #body
            img = v[0,0,j,k]
            img = cut_img(img,5)
            img = misc.imresize(img,(h,h))
            # if j==0: img = 255-misc.imfilter(img,"contour")
            v_new[0,0,j,k] = img

            #hand
            img = v[0,1,j,k]
            img = img[sli,sli]
            img = misc.imresize(img,(h,h))
            v_new[0,1,j,k] = img
    return v_new

def norm_vid(vid):
    vid_s = vid.shape
    vid = vid.reshape(prod(vid_s[:-1]),vid_s[-1])
    cv2.normalize(vid, vid, 0, 255, cv2.NORM_MINMAX)
    vid = vid.reshape(vid_s)
    return vid

def cut_img(img,s):
    if s==0: return img 
    return img[s:-s,s:-s]

def normalize(x):
    _min = x.min()
    return 1.*(x-_min)*255./(x.max()-_min)

def show_img(img, wait=50, resize=True,size=400):
    # cv2.normalize(img,img, 0, 255, cv2.NORM_MINMAX)
    img = img.astype("uint8")
    # print ratio
    if resize:
        if img.shape[0]<img.shape[1]:
            ratio = 1.*img.shape[0]/img.shape[1]
            r = size
            res = (r,int(ratio*r))
        else:
            ratio = 1.*img.shape[1]/img.shape[0]
            r = size
            # print ratio
            res = (int(ratio*r),r)
        img = cv2.resize(img, res, interpolation=cv2.INTER_NEAREST)

    # cv2.namedWindow("Img", cv2.WND_PROP_FULLSCREEN) 
    # cv2.setWindowProperty("Img", cv2.WND_PROP_FULLSCREEN, cv2.cv.CV_WINDOW_FULLSCREEN)
    cv2.imshow("Img", img)
    cv2.waitKey(wait)

def play_vid(vid, wait=50):
    import cv2
    for i,img in enumerate(vid):
        # cv2.normalize(img,img, 0, 255, cv2.NORM_MINMAX)
        img = cv2.resize(img.astype("uint8"), (200,200))
        cv2.imshow("Gesture", img)
        cv2.waitKey(wait)

def load_gzip(path):
    file = gzip.GzipFile(path, 'rb')
    r = load(file)
    file.close()
    return r

if __name__ == '__main__':
    main()