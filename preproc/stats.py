from ChalearnLAPSample import Skeleton, GestureSample
from cPickle import load, dump
from glob import glob
from random import shuffle
import cv2
import os
import shutil
import errno
import time
import gzip
from numpy import *
from numpy import linalg
import sys

#settings
data = "/media/Extern/chalearn2014/train_raw"
dest = "/media/Data/mp/chalearn2014/40x90x90_train"
store_result = True
bg_remove = False
cutout_body = True

show_gray = False
show_depth = False
show_user = False

vid_res = (480, 640) # 640 x 480 video resolution
vid_shape_hand = (40, 90, 90)
vid_shape_body = (40, 90, 90)

#globals
offset = vid_shape_hand[1]/2
n_frames = vid_shape_hand[0]
statw, stato, statn = [], [], []
store_data = [],[],[]
batch_idx = 0


def main():
    from numpy.random import RandomState
    rng = RandomState(1337)

    # get samples
    os.chdir(data)
    samples=glob("*.zip")

    # shuffle or sort
    # samples.sort()
    rng.shuffle(samples)

    # remove previously unzipped files
    for i in set(glob("*")).difference(samples): shutil.rmtree(i) 

    print len(samples), "samples found"

    #start preprocessing
    gather_stats(samples)


def gather_stats(samples):

    for file in samples:
        print "Processing", file 
        smp = GestureSample(data+"/"+file)
        # proc_sample(sample)
        gestures = smp.getGestures()
        for gesture in gestures:
            skelet = []
            id,start,end = gesture
            n_f = n_frames
            n = smp.data['numFrames']
            l = end - start
            statn.append(end - start)
            # start = start + l/2 -n_f/2
            # end = start + n_f
            # if start < 1: start,end = (1,1+n_f)
            # elif end >= n: start,end = (n-1-n_f,n-1)
            # l = n_frames

            # for i,framenum in enumerate(range(start,end)): skelet.append(smp.getSkeleton(framenum))

            # phl, phr, ph, pc = [empty((2,l)) for _ in range(4)]
            # whl, whr, wh, wc = [empty((3,l)) for _ in range(4)]
            # ohl, ohr = [empty((4,l)) for _ in range(2)]

            # for i,skel in enumerate(skelet):
            #     pix = skel.getPixelCoordinates()
            #     world = skel.getWorldCoordinates()
            #     ori = skel.getJoinOrientations()
            #     phl[:,i] = array(pix['HandLeft'])
            #     phr[:,i] = array(pix['HandRight'])
            #     whl[:,i] = array(world['HandLeft'])
            #     whr[:,i] = array(world['HandRight'])
            #     ohl[:,i] = array(ori['HandLeft'])
            #     ohr[:,i] = array(ori['HandRight'])
            #     ph[:,i] = array(pix['Head'])
            #     pc[:,i] = array(pix['HipCenter'])
            #     wh[:,i] = array(world['Head'])
            #     wc[:,i] = array(world['HipCenter'])

            # if count_nonzero(phl) < 10*2: continue

            # phl,phr,ph,pc,whl,whr,wh,wc = [smooth(s) for s in \
            #                                         phl,phr,ph,pc,whl,whr,wh,wc]
            # ohl,ohr = [smooth(s,3) for s in ohl,ohr]

            # phl_y = phl[1][phl[1].nonzero()]
            # phr_y = phr[1][phr[1].nonzero()]

            # hand = "left" if phl_y.mean() < phr_y.mean() else "right"

            # if hand=="left":
            #     # whl[0] = whl[0]*(-1)
            #     traj2D,traj3D,ori = phl, whl, ohl
            # else:
            #     traj2D,traj3D,ori = phr, whr, ohr

            # wheight = array([linalg.norm(wc[:,i]-wh[:,i]) for i in range(l)]).mean()

            # traj3D = (wh-traj3D)/wheight

            # if hand=="left": 
            #     traj3D[0] *=-1
            #     # print traj3D[0].min(), traj3D[0].mean(), traj3D[0].max()

            # statw.append([ [traj3D[0].min(), traj3D[0].max()],
            #             [traj3D[1].min(), traj3D[1].max()],
            #             [traj3D[2].min(), traj3D[2].max()]])
            # stato.append([[ori[0].min(), ori[0].max()],
            #             [ori[1].min(), ori[1].max()],
            #             [ori[2].min(), ori[2].max()],
            #             [ori[3].min(), ori[3].max()]])

            # traj3D,ori = [d.astype("uint8") for d in traj3D,ori]


        report_stats()

def report_stats():
    # stw,sto = [array(i) for i in statw,stato]

    # def report(stat,s):
    #     for i in range(stat.shape[1]):
    #         a,b = stat[:,i,0].mean(), stat[:,i,1].mean()
    #         a -= stat[:,i,0].std()
    #         b += stat[:,i,1].std()
    #         print s+"%i %8.5f %8.5f" % (i, a,b)

    # report(stw,"w")
    # report(sto, "o")
    stn = array(statn) 
    print stn.mean(), median(stn)


def get_data(smp, gesture):
    corrupt = False
    s = []
    id,start,end = gesture

    dv,uv,gv = smp.depth, smp.user, smp.rgb


    d,u,g = [empty((n_frames,)+vid_res+(3,), "uint8") for _ in range(3)]

    total_n_frames = smp.data['numFrames']
    l = end - start
    if l < n_frames:
        start -= (n_frames - l)/2
        if start < 1: start = 1
        end = start + n_frames
        if end >= total_n_frames: 
            end = total_n_frames-1
            start = end - n_frames

    elif l > n_frames:
        start = end - l/2-n_frames/2
        end = start + n_frames

    for v in dv,uv,gv: go_to_frame(v, start)

    for i,framenum in enumerate(range(start,end)):
        s.append(smp.getSkeleton(framenum))
        d[i],u[i],g[i] = [v.read()[1] for v in dv,uv,gv]
    
    d,u,g = [to_grayscale(v) for v in d,u,g]
    # d,u = [to_grayscale(v) for v in d,u]
    u[u<128], u[u>=128] = 0, 1

    if count_nonzero(u)<10: corrupt = True
    return s,d,g,u, corrupt

def proc_user(user):
    user[user==1]=255
    for i,u in enumerate(user):
        u = cv2.medianBlur(u, 3)
        user[i] = u

    user.swapaxes(0,1)
    for i,u in enumerate(user):
        u = cv2.medianBlur(u, 9)
        user[i] = u
    user.swapaxes(0,1)

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

def proc_skelet(skelet):
    corrupt = False
    l = len(skelet)
    phl, phr, ph, pc = [empty((2,l)) for _ in range(4)]
    whl, whr, wh, wc = [empty((3,l)) for _ in range(4)]
    ohl, ohr = [empty((4,l)) for _ in range(2)]

    for i,skel in enumerate(skelet):
        pix = skel.getPixelCoordinates()
        world = skel.getWorldCoordinates()
        ori = skel.getJoinOrientations()
        phl[:,i] = array(pix['HandLeft'])
        phr[:,i] = array(pix['HandRight'])
        whl[:,i] = array(world['HandLeft'])
        whr[:,i] = array(world['HandRight'])
        ohl[:,i] = array(ori['HandLeft'])
        ohr[:,i] = array(ori['HandRight'])
        ph[:,i] = array(pix['Head'])
        pc[:,i] = array(pix['HipCenter'])
        wh[:,i] = array(world['Head'])
        wc[:,i] = array(world['HipCenter'])

    if count_nonzero(phl) < 10*2: 
        corrupt = True
    else:
        phl,phr,ph,pc,whl,whr,wh,wc = [smooth(s) for s in \
                                                phl,phr,ph,pc,whl,whr,wh,wc]
        ohl,ohr = [smooth(s,3) for s in ohl,ohr]

    # phl_ = phl[0][phl[0].nonzero()]
    # phr_ = phr[0][phr[0].nonzero()]
    phl_y = phl[1][phl[1].nonzero()]
    phr_y = phr[1][phr[1].nonzero()]
    # varl, varr = phl_.var(),phr_.var()
    # print varl, varr
    # print phl_y.mean(), phr_y.mean()
    # if abs(varl-varr) < 1000:
    #     hand = "left" if phl_y.mean() < phr_y.mean() else "right"
    # else: hand = "left" if varl > varr else "right"

    hand = "left" if phl_y.mean() < phr_y.mean() else "right"
    # hand = "left" if phl_[1].mean() < phr_[1].mean() else "right"
    print hand
    if hand=="left":
        # whl[0] = whl[0]*(-1)
        traj2D,traj3D,ori = phl, whl, ohl
    else:
        traj2D,traj3D,ori = phr, whr, ohr

    pheight = array([linalg.norm(pc[:,i]-ph[:,i]) for i in range(l)]).mean()
    wheight = array([linalg.norm(wc[:,i]-wh[:,i]) for i in range(l)]).mean()
    center = pc.mean(1)

    traj3D = (wh-traj3D)/wheight

    if hand=="left": 
        traj3D[0] = traj3D[0]*(-1)
        print traj3D[0].min(), traj3D[0].mean(), traj3D[0].max()

    traj3D[0]    = normalize(traj3D[0],-0.04, 0.58)
    traj3D[1]    = normalize(traj3D[1], 0.15, 1.53)
    traj3D[2]    = normalize(traj3D[2],-0.04, 0.72)
    ori[0]       = normalize(    ori[0], 0.36, 1.00)
    ori[1]          = normalize(    ori[1],-0.26, 0.19)
    ori[2]          = normalize(    ori[2],-0.10, 0.12)
    ori[3]          = normalize(    ori[3],-0.31, 0.46)

    """
    w0 -0.06244  0.61260
    w1  0.10840  1.60145
    w2 -0.09836  0.76818
    o0  0.30971  1.00535
    o1 -0.27595  0.19067
    o2 -0.10569  0.12660
    o3 -0.32591  0.48749
    """

    # statw.append([ [traj3D[0].min(), traj3D[0].max()],
    #             [traj3D[1].min(), traj3D[1].max()],
    #             [traj3D[2].min(), traj3D[2].max()]])
    # stato.append([[ori[0].min(), ori[0].max()],
    #             [ori[1].min(), ori[1].max()],
    #             [ori[2].min(), ori[2].max()],
    #             [ori[3].min(), ori[3].max()]])

    traj3D,ori = [d.astype("uint8") for d in traj3D,ori]

    return (traj2D,traj3D,ori,pheight,hand,center), corrupt


def normalize(x, old_min, old_max, new_min=0, new_max=255):
    """ Normalize numpy array """
    x = clip(x,old_min, old_max)
    return 1.*(x-old_min)*(new_max-new_min)/(old_max-old_min)+new_min


def proc_depth(depth, user, user_o, skelet):
    # settings
    thresh_noise = 220
    scaler = 4

    corrupt = False
    traj2D,traj3D,ori,pheight,hand,center = skelet

    #stats
    user_depth = depth[user_o==1]
    med = median(user_depth)
    med = 255 - med
    std = user_depth.std()


    depth_b = cut_body(depth.copy(), center, pheight)
    user_b = cut_body(user.copy(), center, pheight)

    depth_h = cut_hand(depth.copy(), traj2D, hand)
    user_h = cut_hand(user.copy(), traj2D, hand)

    new_depth = empty((2,)+vid_shape_hand, dtype="uint8")

    for i,(depth,user) in enumerate(((depth_b,user_b),(depth_h,user_h))):

        # invert grayscale: 0 -> background
        depth = 255 - depth


        nuser_depth = depth[user==0]
        nuser_depth[nuser_depth>thresh_noise] = 0
        depth[user==0] = nuser_depth
        depth[depth>thresh_noise] = 0
        #depth = inpaint(depth, thresh_noise)
        thresh_depth = med-4*std
        depth[depth<thresh_depth] = thresh_depth   
        depth = depth-thresh_depth
        depth = clip(depth*scaler, 0, 255)
        depth = depth.astype("uint8")
        depth = medianblur(depth)

        new_depth[i] = depth

    depth = new_depth
    return depth.astype("uint8"), corrupt


def proc_gray(gray, user, skelet):
    krn = 9

    corrupt = False
    traj2D,traj3D,ori,pheight,hand,center = skelet

    gray_b = cut_body(gray, center, pheight)
    user_b = cut_body(user, center, pheight)

    gray_h = cut_hand(gray, traj2D, hand)
    user_h = cut_hand(user, traj2D, hand)

    # -----------------USER-------------------------------
    # if bg_remove:
    #     shp = user.shape
    #     user= user.reshape((prod(shp[:-1]),shp[-1]))
    #     user[user==1]=255
    #     # user = cv2.medianBlur(user.astype("uint8"),5)
    #     user = cv2.GaussianBlur(user.astype("uint8"),(7,7),0)
    #     user[user<=0],user[user>0]=0,1
    #     user = user.reshape(shp)
    #     shp = gray.shape
    # -----------------USER-------------------------------
    new_gray = empty((2,)+vid_shape_hand, dtype="uint8")

    for j,(gray,user) in enumerate(((gray_b,user_b),(gray_h,user_h))):

        # -----------------LCN-------------------------------
        gray = gray.astype("float")/255.
        # new_gray = empty(gray.shape)
        mean_img = gray.mean(axis=0)
        # show_img(mean_img)
        for i,g in enumerate(gray):

            nom = g - cv2.GaussianBlur(g, (krn,krn),0)
            den = sqrt(cv2.GaussianBlur(g**2, (krn,krn),0))
            den = maximum(1e-4, den, mean_img)
            g = nom/den

            # global contrast
            mean = g.mean()
            new_g = g-mean
            norm = sqrt((new_g ** 2).sum(axis=1))
            g /= norm[:,None]


            if bg_remove: 
                u = user[i]
                g[u==0] = 0
            gray[i] = g
        # -----------------LCN-------------------------------


        gray = norm_vid(gray)

        new_gray[j] = gray
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
        file_name = "batch_"+str(len(l))+"_"+str(batch_idx)+".zip"
        if store_result:

            file = gzip.GzipFile(file_name, 'wb')
            dump((v,s,l), file, -1)
            file.close()

            # dump((t,o,d,g,l), open(file_name,"wb"),-1)
        print file_name
        batch_idx += 1
        store_data = [],[],[]



def cut_hand(vid, traj, hand):
    # new_vid = empty((2,vid.shape[0],offset*2,offset*2), "uint8")
    new_vid = empty(vid_shape_hand, "uint8")
    for i,img in enumerate(vid):
        img = cut_hand_img(img, traj[:,i])
        if hand == "left": img = cv2.flip(img,1)
        new_vid[i] = img

    return new_vid


def cut_hand_img(img, center):
    c = center.astype("int")

    x = (c[0]-offset, c[0]+offset)
    y = (c[1]-offset, c[1]+offset)
    x,y = fit_screen(x,y)

    # cut out hand    
    img = img[y[0]:y[1],x[0]:x[1]]
    return img


#cutout body, center = hipcenter
def cut_body(vid, center, height):
    c = center
    h = height
    y = c[1]*1.1
    l = int(h*1.3 + (y-c[1]))
    x = c[0]-l/2

    y = (y-l,y)
    x = (x,x+l)
    x,y = fit_screen(x,y)

    vid = vid[:,y[0]:y[1],x[0]:x[1]]

    body = empty(vid_shape_body, dtype=vid.dtype)
    for i,u in enumerate(vid):
        body[i] = cv2.resize(u,vid_shape_body[1:],
            interpolation=cv2.INTER_LINEAR)

    return body


def fit_screen(x,y):
    l = x[1]-x[0]
    r = vid_res

    assert l == y[1]-y[0]

    if y[0] < 0: y=(0,l)
    elif y[1] > r[0]: y = (r[0]-l,r[0])

    if x[0] < 0: x=(0,l)
    elif x[1] > r[1]: x = (r[1]-l,r[1])

    return x,y


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
    if window == 'flat': #moving average
            w=ones(window_len,'d')
    else:  
            w=eval(window+'(window_len)')
    y=convolve(w/w.sum(),s,mode='same')
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


if __name__ == '__main__': main()