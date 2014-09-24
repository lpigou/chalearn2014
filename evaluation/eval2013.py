import sys
sys.path.append('../preproc')
# sys.path.append('..')
import eval_model
from glob import glob
from gzip import GzipFile
from cPickle import load
import os
import zipfile
import shutil
import time
import cv2
from numpy import *
import csv
import preproc as pp
from scipy import ndimage
from scipy import misc
from numpy.random import RandomState
from multiprocessing import Process, Queue


print "step5max0.89"

rng = RandomState(1337)
vid_res = (480, 640) # 640 x 480 video resolution
vid_shape = (32, 64, 64)
cpu = True

data = "/media/Data/mp/chalearn2013/test"
# data = "/media/lio/64EE5F7C8CC54BFB/chalearn2014/valid"
dst = "/home/lio/Dropbox/MP/chalearn2014/evaluation/results/2013step5max0.89.csv"

pp.make_sure_path_exists(dst)
step = 5
threshold = 0.89
n_f = 32
pred_file_name = ''
h = vid_shape[-1]
q = Queue(10)
q2 = Queue(20)

def eval():

    files = glob(data+"/"+"*.zip")
    files.sort()

    q2.put("Id,Sequence")

    print len(files), "found"

    for fileName in files[:]:

        num = fileName.split('/')
        # pred_file_name = pred_file_name[-1].replace(".zip","_prediction.csv")
        num = num[-1].replace(".zip","")[-4:]
        q2.put("\n"+str(num))
        print fileName

        smp = pp.GestureSample(fileName)
        s_time = time.time()
        n = smp.data['numFrames']
        dv,uv,gv = smp.depth, smp.user, smp.rgb


        cur_fr = 1
        new_shape = (step,128,128)

        s = []
        d,u,g = [empty((n_f,)+vid_res+(3,), "uint8") for _ in range(3)]
        for v in dv,uv,gv: pp.go_to_frame(v, cur_fr)
        for i,fr in enumerate(range(cur_fr,cur_fr+n_f)):
            s.append(smp.getSkeleton(fr))
            d[i],u[i],g[i] = [v.read()[1] for v in dv,uv,gv]

        d,u,g = [pp.to_grayscale(v) for v in d,u,g]
        u[u<128], u[u>=128] = 0, 1
        depth,user,gray,skelet = d,u,g,s
        user_o = user.copy()
        depth_o = depth.copy()
        gray_o = gray.copy()
        user_depth = depth_o[user_o==1]
        skelet,c =pp.proc_skelet(array(skelet).copy())
        user = pp.proc_user(user)


        _,depth,c = pp.proc_depth(depth.copy(), user.copy(), user_o, array(skelet).copy())
        gray,c = pp.proc_gray(gray.copy(), user,  array(skelet).copy()) #user.copy!!!!!!!!!!!!!!!!!!!

        cur_fr += n_f

        while cur_fr+step<n:
            sn=[]
            dn,un,gn = [empty((step,)+vid_res+(3,), "uint8") for _ in range(3)]
            for i,fr in enumerate(range(cur_fr,cur_fr+step)):
                sn.append(smp.getSkeleton(fr))
                dn[i],un[i],gn[i] = [v.read()[1] for v in dv,uv,gv]

            dn,un,gn = [pp.to_grayscale(v) for v in dn,un,gn]
            un[un<128], un[un>=128] = 0,1

            s = s[step:] + sn
            skelet,c =pp.proc_skelet(s,_3D=False)

            len_dump = len(depth_o[:step][user_o[:step]==1])
            un_d = dn[un==1]

            user_o[:-step]=user_o[step:]
            user_o[-step:] = un.copy()
            un = pp.proc_user(un,3)

            user[:-step]=user[step:]
            user[-step:] = un.copy()

            depth_o[:-step]=depth_o[step:]
            depth_o[-step:] = dn.copy()
            gray_o[:-step]=gray_o[step:]
            gray_o[-step:] = gn.copy()

            _,depth,c = pp.proc_depth(depth_o.copy(), user.copy(), user_o, skelet)
            gray,c = pp.proc_gray(gray_o.copy(), user,  skelet)
            traj2D,traj3D,ori,pheight,hand,center = skelet

            video = empty((1,2,)+gray.shape,dtype="uint8")
            video[0,0] = gray.copy()
            video[0,1] = depth.copy()
            video = video.swapaxes(1,2) #(body-hand,gray-depth,fr,h,w)

            # ---------------------------AUGMENT--------------------------------------
            v_new = empty((1,2,2)+vid_shape,dtype="uint8")
            # p = pheight
            ratio = 0.25
            for i in xrange(video.shape[0]): #batch

                if pheight < 10: pheight = 100
                scale = ratio+randi(2)/100.
                ofs = pheight*scale
                mid =  video.shape[-1]/2.
                sli = None
                if ofs < mid:
                    start = int(round(mid-ofs))
                    end = int(round(mid+ofs))
                    sli = slice(start,end)

                for j in xrange(video.shape[2]): #maps
                    for k in xrange(video.shape[3]): #frames
                        #body
                        img = video[i,0,j,k]
                        img = cut_img(img,5)
                        img = misc.imresize(img,(h,h))
                        v_new[i,0,j,k] = img

                        #hand
                        img = video[i,1,j,k]
                        img = img[sli,sli]
                        img = misc.imresize(img,(h,h))
                        v_new[i,1,j,k] = img
            # ---------------------------AUGMENT--------------------------------------

            q.put((v_new,cur_fr,n, fileName))

            cur_fr += step

def pred_loop():
    evalu_model, x_ = eval_model.build(cpu)

    global pred_file_name

    while True:

        vid, cur_fr,n,fileName = q.get()
        # pred_file_name = fileName.split('/')
        # # pred_file_name = pred_file_name[-1].replace(".zip","_prediction.csv")
        # pred_file_name = pred_file_name[-1].replace(".zip","")[-4:]
        x_.set_value(vid.astype("float32"),borrow=True)
        pred = evalu_model()[0][0]
        pred_p = pred.max()
        pred_idx = pred.argmax()+1
        fr_start = cur_fr+step-n_f
        fr_end = cur_fr+step
        predict(pred_idx,pred_p,fr_start)
        if cur_fr+2*step>=n:
            reinit()


fr_med = 37
last_pred = -1
pred_count = 0
start_frs = []
pred_ps = []
detected = False
def predict(pred_idx,pred_p,fr_start):
    global last_pred,pred_count,detected,start_frs,pred_ps
    if pred_p < 0.8 or pred_idx==21:
        reinit()
        return
    if last_pred != pred_idx:
        reinit()
        last_pred = pred_idx
        pred_count += 1
        start_frs.append(fr_start)
        pred_ps.append(pred_p)
        return
    if last_pred == pred_idx:
        pred_count += 1
        start_frs.append(fr_start)
        pred_ps.append(pred_p)
        if not detected and (pred_count-1)*step>=10:
            detected = True

def play_vid(vid, wait=50):
    import cv2
    for i,img in enumerate(vid):
        cv2.normalize(img,img, 0, 255, cv2.NORM_MINMAX)
        img = cv2.resize(img.astype("uint8"), (200,200))
        cv2.imshow("Gesture", img)
        cv2.waitKey(wait)

def reinit():
    global last_pred,pred_count,detected,start_frs,pred_ps

    if detected:
        pred_ps = array(pred_ps)
        m = pred_ps.max()
        if m > threshold:
            ind = where(pred_ps==m)[0]
            ind = int(ind[int(len(ind)/2)])

            fr_st = start_frs[ind] + int((n_f-fr_med)/2)
            fr_end = fr_st+fr_med
            _s = ",%i"%(last_pred)
            q2.put(_s)


    last_pred = -1
    pred_count = 0
    start_frs = []
    pred_ps = []
    detected = False

def write_loop():
    while True:
        s = q2.get()
        with open(dst,"a") as f: f.write(s)


def cut_img(img,s):
    if s==0: return img 
    return img[s:-s,s:-s]

def randi(i): return rng.randint(-i,i)

if __name__ == '__main__': 
    p_ = Process(target=eval, args=())
    p_.daemon=True
    p_.start()
    p2 = Process(target=write_loop, args=())
    p2.daemon=True
    p2.start()
    pred_loop()
    p_.terminate()
    p2.terminate()
        
