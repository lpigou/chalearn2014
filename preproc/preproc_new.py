"""
vid,skelet,labels = load()

vid: (100,2,2,40,90,90): (batch,gray-depth,body-hand,frames,height,width)

traj,ori,pheight = skelet 

traj: (100,3,40): (batch,x-y-z,frames)
ori:    (100,4,40): (batch,x-y-z-w,frames)
pheight: (100,): height of hipcenter to head in pixels

labels: (100,)
"""

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
from itertools import tee, islice
from numpy import *
from numpy import linalg
from numpy.random import RandomState

#settings
# data = "/media/Data/mp/chalearn2014/train_raw"
# dest = "/media/Data/mp/chalearn2014/40x90x90_train_21lbls"
data = "/media/lio/64EE5F7C8CC54BFB/chalearn2014/valid"
dest = "/media/lio/64EE5F7C8CC54BFB/chalearn2014/21lbl_32x128x128/valid2"
file_valid_samps = "samples_bg1.txt"
store_result = True
bg_remove = False
norm_gray = True

show_gray = False
show_depth = False
show_user = False

vid_res = (480, 640) # 640 x 480 video resolution
vid_shape_hand = (32, 128, 128)
vid_shape_body = (32, 128, 128)

#globals
offset = vid_shape_hand[1]/2
n_frames = vid_shape_hand[0]
store_data = [],[],[]
batch_idx = 0
n_div, p_i = None, None
valid_samps = None

def main(args):
    global n_div, p_i, valid_samps
    rng = RandomState(1337)
    if len(args) == 2:
        n_div,  p_i =int(args[0]),int(args[1])
    else: n_div,p_i = 1,0


    # get samples
    #valid_samps = open(file_valid_samps,"rb")
    #valid_samps = (valid_samps.read()).split("\n")
    os.chdir(data)
    samples=glob("*.zip")
    # samples = list(set(samples).difference(valid_samps))
    # samples = valid_samps
    samples.sort()

    def partition(iterable,parts):
        return [list(islice(it, i, None, parts)) for i, it in enumerate(tee(iterable, parts))]

    samples = partition(samples,n_div)[p_i]
    # shuffle or sort
    samples.sort()
    # rng.shuffle(samples)
    # shuffle(samples)

    # remove previously unzipped files
    # for i in set(glob("*")).difference(samples): shutil.rmtree(i) 

    print len(samples), "samples found"

    #start preprocessing
    preprocess(samples)

# @profile
def preprocess(samples):

    for file in samples:
        print "Processing", file 
        sample = GestureSample(data+"/"+file)
        # proc_sample(sample)
        gestures = sample.getGestures()
        for i in range(len(gestures)-1):
            end_prev = gestures[i][2]
            st_next = gestures[i+1][1]
            l = st_next-end_prev
            if l > n_frames:
                start = end_prev + int((l-n_frames)/2.)
                end = start + n_frames
                gestures.append([21,start,end])
                break
        gestures.sort(reverse=True)
        # print gestures
        for gesture in gestures:
            skelet, depth, gray, user, c = get_data(sample, gesture)
            if c: print 'corrupt'; continue

            user_o = user.copy()

            # preprocess
            skelet,c = proc_skelet(skelet)
            if c: print 'corrupt'; continue
            user = proc_user(user)
            user_new, depth,c = proc_depth(depth, user, user_o, skelet)
            if c: print 'corrupt'; continue
            gray,c = proc_gray(gray, user,  skelet)
            if c: print 'corrupt'; continue

            user = user_new

            if show_depth: play_vid(depth,norm=False)
            if show_gray: play_vid(gray, norm=False)
            if show_user: play_vid(user,norm=True)

            # user_new = user_new.astype("bool")

            traj2D,traj3D,ori,pheight,hand,center = skelet
            skelet = traj3D,ori,pheight

            assert user.dtype==gray.dtype==depth.dtype==traj3D.dtype==ori.dtype=="uint8"
            assert user.shape==gray.shape==depth.shape==(2,)+vid_shape_hand
            assert traj3D.shape[1]==ori.shape[1]==n_frames

            video = empty((3,)+gray.shape,dtype="uint8")
            video[0],video[1],video[2] = gray,depth,user
            store_preproc(video,skelet,gesture[0])

        # dump_data(file)
    dump_last_data()
    print 'Process',p_i,'finished'


def get_data(smp, gesture):
    corrupt = False
    s = []
    id,start,end = gesture

    dv,uv,gv = smp.depth, smp.user, smp.rgb

    n_f = n_frames
    d,u,g = [empty((n_f,)+vid_res+(3,), "uint8") for _ in range(3)]

    n = smp.data['numFrames']
    l = end - start

    start = start + l/2 -n_f/2
    end = start + n_f

    if start < 1: start,end = (1,1+n_f)
    elif end >= n: start,end = (n-1-n_f,n-1)

    for v in dv,uv,gv: go_to_frame(v, start)

    for i,framenum in enumerate(range(start,end)):
        s.append(smp.getSkeleton(framenum))
        d[i],u[i],g[i] = [v.read()[1] for v in dv,uv,gv]
    
    d,u,g = [to_grayscale(v) for v in d,u,g]
    # d,u = [to_grayscale(v) for v in d,u]
    u[u<128], u[u>=128] = 0, 1

    if count_nonzero(u)<10: corrupt = True
    return s,d,g,u, corrupt


def proc_user(user,krn=11):
    user[user==1]=255
    for i,u in enumerate(user):
        # u = cv2.medianBlur(u, 3)
        u = cv2.medianBlur(u, krn)
        user[i] = u

    # user = user.swapaxes(0,1)
    # for i,u in enumerate(user):
    #     u = cv2.medianBlur(u, 9)
    #     user[i] = u
    # user = user.swapaxes(0,1)
    #---------------------CONTOUR--------------------------------------------
    # for i,u in enumerate(user):
    #     # u = cv2.medianBlur(u, 3)
    #     contours, hierarchy = cv2.findContours(u.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    #     im = zeros(u.shape)
    #     # cnt = contours[4]
    #     maxl = 0
    #     biggest_cnt = None
    #     for cnt in contours:
    #         if len(cnt) > maxl:
    #             maxl = len(cnt)
    #             biggest_cnt = cnt

    #     cv2.drawContours(im,[biggest_cnt],0,255,-1)

    #     user[i] = im
    #---------------------CONTOUR--------------------------------------------
    user[user>0] = 1
    return user

# @profile
def proc_skelet(skelet, _3D=True):
    corrupt = False
    traj2D,traj3D,ori,pheight,hand,center = None,None,None,None,None,None
    l = len(skelet)
    phl, phr, ph, pc = [empty((2,l)) for _ in range(4)]
    if _3D:
        whl, whr, wh, wc = [empty((3,l)) for _ in range(4)]
        ohl, ohr = [empty((4,l)) for _ in range(2)]

    for i,skel in enumerate(skelet):
        pix = skel.getPixelCoordinates()
        if _3D:
            world = skel.getWorldCoordinates()
            ori = skel.getJoinOrientations()
            whl[:,i] = array(world['HandLeft'])
            whr[:,i] = array(world['HandRight'])
            ohl[:,i] = array(ori['HandLeft'])
            ohr[:,i] = array(ori['HandRight'])
            wh[:,i] = array(world['Head'])
            wc[:,i] = array(world['HipCenter'])
        phl[:,i] = array(pix['HandLeft'])
        phr[:,i] = array(pix['HandRight'])
        ph[:,i] = array(pix['Head'])
        pc[:,i] = array(pix['HipCenter'])

    if count_nonzero(phl) < 10*2: 
        corrupt = True
    # elif _3D:
    #     phl,phr,ph,pc,whl,whr,wh,wc = [smooth(s) for s in \
    #                                             phl,phr,ph,pc,whl,whr,wh,wc]
    #     ohl,ohr = [smooth(s,3) for s in ohl,ohr]
    # else:
    #     phl,phr,ph,pc = [smooth(s) for s in phl,phr,ph,pc]

    phl_y = phl[1][phl[1].nonzero()]
    phr_y = phr[1][phr[1].nonzero()]

    hand = "left" if phl_y.mean() < phr_y.mean() else "right"

    if hand=="left": 
        traj2D=phl
        if _3D: traj3D,ori = whl, ohl
    else: 
        traj2D = phr
        if _3D: traj3D,ori = whr, ohr

    pheight = empty((l,),dtype="float32")
    # wheight = empty((l,),dtype="float32")
    for i in xrange(l): pheight[i] = linalg.norm(pc[:,i]-ph[:,i])
    #     wheight[i] = linalg.norm(wc[:,i]-wh[:,i])
    pheight = pheight.mean()
    # wheight = wheight.mean()
    # pheight = array([linalg.norm(pc[:,i]-ph[:,i]) for i in range(l)]).mean()
    if _3D:
        wheight = array([linalg.norm(wc[:,i]-wh[:,i]) for i in range(l)]).mean()
        traj3D = (wh-traj3D)/wheight
        if hand=="left": traj3D[0] *= -1
        traj3D[0]    = normalize(traj3D[0],-0.06244 , 0.61260)
        traj3D[1]    = normalize(traj3D[1], 0.10840 , 1.60145)
        traj3D[2]    = normalize(traj3D[2],-0.09836 , 0.76818)
        ori[0]       = normalize(ori[0]   , 0.30971 , 1.00535)
        ori[1]       = normalize(ori[1]   ,-0.27595 , 0.19067)
        ori[2]       = normalize(ori[2]   ,-0.10569 , 0.12660)
        ori[3]       = normalize(ori[3]   ,-0.32591 , 0.48749)

        traj3D,ori = [d.astype("uint8") for d in traj3D,ori]
    
    center = pc

    return (traj2D,traj3D,ori,pheight,hand,center), corrupt


def proc_depth(depth, user, user_o, skelet):
    # settings
    thresh_noise = 200
    scaler = 4

    corrupt = False
    traj2D,traj3D,ori,pheight,hand,center = skelet

    #stats
    depth = 255 - depth
    user_depth = depth[user_o==1]
    # print histogram(user_depth,100)
    # thresh_noise = user_depth.max()
    med = average(user_depth)
    # med = 255 - med
    std = user_depth.std()
    depth_b = cut_body(depth.copy(), center, pheight, hand)
    user_b = cut_body(user.copy(), center, pheight, hand)

    depth_h = cut_hand(depth.copy(), traj2D, hand)
    user_h = cut_hand(user.copy(), traj2D, hand)

    new_depth = empty((2,)+vid_shape_hand, dtype="uint8")

    for i,(depth,user) in enumerate(((depth_b,user_b),(depth_h,user_h))):


        # nuser_depth = depth[user==0]
        # nuser_depth[nuser_depth>thresh_noise] = 0
        # depth[user==0] = nuser_depth
        # depth[user==0] = 0

        depth[depth>thresh_noise] = 0
        # depth = inpaint(depth, thresh_noise)

        thresh_depth = med-3*std
        # print med-3*std
        # thresh_depth = 100
        # depth[depth<thresh_depth] = thresh_depth
        depth = depth-thresh_depth
        depth = clip(depth*scaler, 0, 255)
        # depth = depth - depth.mean()
        # depth = norm_vid(depth)
        depth = depth.astype("uint8")
        depth = medianblur(depth)
        new_depth[i] = depth

    depth = new_depth

    new_user = empty((2,)+vid_shape_hand, dtype="uint8")
    new_user[0] = user_b
    new_user[1] = user_h

    return new_user, depth.astype("uint8"), corrupt


def proc_gray(gray, user, skelet):
    # krn = 9

    corrupt = False
    traj2D,traj3D,ori,pheight,hand,center = skelet

    # gray[user==0] = 255
    # gray = inpaint(gray, 255)


    gray_b = cut_body(gray.copy(), center, pheight, hand)
    # user_b = cut_body(user.copy(), center, pheight, hand)

    gray_h = cut_hand(gray.copy(), traj2D, hand)
    # user_h = cut_hand(user.copy(), traj2D, hand)

    # new_gray = empty((2,)+vid_shape_hand, dtype="uint8")

    # for j,(gray,user) in enumerate(((gray_b,user_b),(gray_h,user_h))):
    #     gray = gray.astype("float")/255.

    #     # mean denominator for local contrast normalisation
    #     den = empty(gray.shape, dtype=gray.dtype)
    #     for i,g in enumerate(gray):
    #         den[i] = sqrt(cv2.GaussianBlur(g**2, (krn,krn),0))
    #     mean_den = den.mean(axis=0)

    #     for i,g in enumerate(gray):
    #         g = local_contrast_normalisation(g, mean_den, krn)
    #         g = global_contrast_normalisation(g)

    #         if bg_remove: g[user[i]==0] = 0
    #         gray[i] = g
    #     # print gray.min(), gray.mean(), gray.max()
    #     if norm_gray: gray = norm_vid(gray)
    #     new_gray[j] = gray

    # gray = new_gray


    new_gray = empty((2,)+vid_shape_hand, dtype="uint8")
    new_gray[0] = gray_b
    new_gray[1] = gray_h
    gray = new_gray

    return gray.astype("uint8"), corrupt


def store_preproc(video,skelet,label):
    global store_data, batch_idx
    v,s,l = store_data
    new_dat = video,skelet,label
    for i,dat in enumerate([v,s,l]): dat.append(new_dat[i])

    if len(l) == 100:
        make_sure_path_exists(dest)
        os.chdir(dest)
        file_name = "batch_"+str(len(l))+"_"+str(p_i)+"_"+str(batch_idx)+".zip"
        if store_result:

            v,l = [array(i,dtype="uint8") for i in v,l]
            v = v.swapaxes(1,2) #(batch,body-hand,gray-depth,fr,h,w)
            s = array(s).swapaxes(0,1)
            traj,ori,pheight = s
            traj,ori,pheight = [array(list(i),dtype="uint8") for i in traj,ori,pheight]
            s = traj,ori,pheight

            file = gzip.GzipFile(file_name, 'wb')
            dump((v,s,l), file, -1)
            file.close()

        print file_name
        batch_idx += 1
        store_data = [],[],[]

def dump_data(file):
    global store_data, batch_idx
    v,s,l = store_data
    if len(l) == 0: return
    make_sure_path_exists(dest)
    os.chdir(dest)
    file_name = file.replace(".zip","_pre.zip")
    if store_result:

        v,l = [array(i,dtype="uint8") for i in v,l]
        v = v.swapaxes(1,2) #(batch,body-hand,gray-depth-user,fr,h,w)
        s = array(s).swapaxes(0,1)
        traj,ori,pheight = s
        traj,ori,pheight = [array(list(i),dtype="uint8") for i in traj,ori,pheight]
        s = traj,ori,pheight

        file = gzip.GzipFile(file_name, 'wb')
        dump((v,s,l), file, -1)
        file.close()

    print file_name
    batch_idx += 1
    store_data = [],[],[]


def dump_last_data():
    global store_data, batch_idx
    v,s,l = store_data
    if len(l) > 0:
        os.chdir(dest)
        file_name = "batch_"+str(len(l))+"_"+str(p_i)+"_"+str(batch_idx)+".zip"
        if store_result:

            v,l = [array(i,dtype="uint8") for i in v,l]
            v = v.swapaxes(1,2) #(batch,body-hand,gray-depth,fr,h,w)
            s = array(s).swapaxes(0,1)
            traj,ori,pheight = s
            traj,ori,pheight = [array(list(i),dtype="uint8") for i in traj,ori,pheight]
            s = traj,ori,pheight

            file = gzip.GzipFile(file_name, 'wb')
            dump((v,s,l), file, -1)
            file.close()

        print file_name


def cut_hand(vid, traj, hand,shape=None):
    # new_vid = empty((2,vid.shape[0],offset*2,offset*2), "uint8")
    if shape:new_vid = empty(shape, "uint8")
    else: new_vid = empty(vid_shape_hand, "uint8")
    for i,img in enumerate(vid):
        img = cut_hand_img(img, traj[:,i])
        if hand == "left": 
            # print "left"
            # if random.randint(1,100)==1: print traj[:,i]
            img = cv2.flip(img,1)
        new_vid[i] = img

    return new_vid


def cut_hand_img(img, center):
    c = center.round().astype("int")

    x = (c[0]-offset, c[0]+offset)
    y = (c[1]-offset, c[1]+offset)
    x,y = fit_screen(x,y)

    # cut out hand    
    img = img[y[0]:y[1],x[0]:x[1]]
    return img


#cutout body, center = hipcenter
def cut_body(vid, center, height, hand,shape=None):
    c = center
    h = height
    if h ==0: h=150

    body = empty(vid_shape_body, dtype=vid.dtype)
    for i in xrange(vid.shape[0]):
        if c[0,i]==0 or c[1,i]==0:
            c[:,i] = array([320, 240])

        y = int(round(c[1,i]+h*0.7))
        l = int(round(h*1.4 + (y-c[1,i])))
        x =int(round(c[0,i]-l/2))

        y = (y-l,y)
        x = (x,x+l)
        x,y = fit_screen(x,y)

        img = vid[i,y[0]:y[1],x[0]:x[1]]
        if hand == "left": 
            img = cv2.flip(img,1)
        img = cv2.resize(img,vid_shape_body[1:], interpolation=cv2.INTER_LINEAR)
        body[i] = img

    # if c[0]==0 or c[1]==0 or h==0:
    #     c = array([320, 240])
    #     h = 150

    # y = int(round(c[1]*1.1))
    # l = int(round(h*1.3 + (y-c[1])))
    # x =int(round(c[0]-l/2))

    # y = (y-l,y)
    # x = (x,x+l)
    # x,y = fit_screen(x,y)

    # vid = vid[:,y[0]:y[1],x[0]:x[1]]
    # if hand == "left": 
    #     for i,img in enumerate(vid): vid[i] = cv2.flip(img,1)

    # if shape: body = empty(shape, dtype=vid.dtype)
    # else: body = empty(vid_shape_body, dtype=vid.dtype)
    # for i,u in enumerate(vid):
    #     body[i] = cv2.resize(u,vid_shape_body[1:],
    #         interpolation=cv2.INTER_LINEAR)

    return body


def fit_screen(x,y):
    l = x[1]-x[0]
    r = vid_res

    if not l == y[1]-y[0]:
        print l, x, y
        raise Exception, "l != y[1]-y[0]"

    if y[0] < 0: 
        y=(0,l)
    elif y[1] > r[0]: 
        y = (r[0]-l,r[0])

    if x[0] < 0: 
        x=(0,l)
    elif x[1] > r[1]: 
        x = (r[1]-l,r[1])

    return x,y


def local_contrast_normalisation(img, mean_den, krnsize):
    krn = (krnsize,krnsize)
    nom = img - cv2.GaussianBlur(img, krn,0)
    den = sqrt(cv2.GaussianBlur(img**2, krn,0))
    den = maximum(1e-4, den, mean_den)
    img = nom/den
    return img


def global_contrast_normalisation(img):
    new_img = img-img.mean()
    # new_img = img.copy()
    norm = sqrt((new_img ** 2).sum(axis=1))
    # norm = sqrt(new_img.var(axis=1))
    new_img /= norm[:,None]
    return new_img


def normalize(x, old_min, old_max, new_min=0, new_max=255):
    """ Normalize numpy array """
    x = clip(x,old_min, old_max)
    return 1.*(x-old_min)*(new_max-new_min)/(old_max-old_min)+new_min


def go_to_frame(vid, frame): 
    vid.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, frame-1)


def to_grayscale(v):
    s = v.shape
    v = v.reshape(prod(s[:-2]),s[-2], s[-1])
    v = cv2.cvtColor(v,cv2.cv.CV_RGB2GRAY)
    return v.reshape(s[:-1])


def medianblur(vid, ksize=3):
    for i,img in enumerate(vid): vid[i] = cv2.medianBlur(img, ksize)
    return vid


def norm_vid(vid):
    vid_s = vid.shape
    vid = vid.reshape(prod(vid_s[:-1]),vid_s[-1])
    cv2.normalize(vid, vid, 0, 255, cv2.NORM_MINMAX)
    vid = vid.reshape(vid_s)
    return vid


def norm_img(img):
    cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)
    return img


def inpaint(vid, threshold):
    for i,img in enumerate(vid):
        mask = img.copy()
        mask[mask<threshold] = 0
        cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA, img) 
        img[img>threshold] = 0
        vid[i] = img
    return vid

w_smooth=ones(5,'d')/5.

def smooth(x,window_len=5,window='flat'):
    if x.ndim > 2:
            raise ValueError, "smooth only accepts 1 or 2 dimension arrays."
    if x.ndim ==2 :
        for i in range(x.shape[0]):
            x[i] = smooth(x[i],window_len,window)
        return x
    if x.shape[0] < window_len:
            raise ValueError, "Input vector needs to be bigger than window size."
    if window_len<3:
            return x
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
            raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"
    s=r_[2*x[0]-x[window_len-1::-1],x,2*x[-1]-x[-1:-window_len:-1]]
    # if window == 'flat': #moving average
    #         w=ones(window_len,'d')
    # else:  
    #         w=eval(window+'(window_len)')
    y=convolve(w_smooth,s,mode='same')
    return y[window_len:-window_len+1]


def make_sure_path_exists(path):
    """Try to create the directory, but if it already exist we ignore the error"""
    try: os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST: raise


def edge_detect(img):
    """ Edge detector """
    cv2.equalizeHist(img, img)
    img = cv2.Canny(img, img.mean()*0.66, img.mean()*1.33, apertureSize=3)
    # img[img != 0] = 255
    img = norm_img(img)
    return img


def play_vid(vid, wait=1000/20, norm=True):
    if isinstance(vid, list):
        for img0,img1 in zip(vid[0],vid[1]): 
            show_imgs([img0,img1], wait, norm)
    elif vid.ndim == 4:
        vid = vid.swapaxes(0,1)
        for imgs in vid: show_imgs(list(imgs), wait, norm)
    else: 
        for img in vid: show_img(img, wait, norm)
     

def show_imgs(imgs, wait=0, norm=True):
    if norm:
        for i, img in enumerate(imgs):
            imgs[i] = norm_img(img)

    img = hstack(imgs)
    show_img(img, wait, False)


def show_img(img, wait=0, norm=True):
    ratio = 1.*img.shape[1]/img.shape[0]
    if norm: img = norm_img(img)
    img = img.astype("uint8")
    size=200 if img.shape[0]<200 else 400
    img = cv2.resize(img, (int(size*ratio), size))
    cv2.imshow("Video", img)
    cv2.waitKey(wait)


if __name__ == '__main__': 
    try:
        main(sys.argv[1:])
    except Exception as e:
        print e
