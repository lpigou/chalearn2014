import os
import sys
import cv2, cv
from glob import glob
from gzip import GzipFile
from cPickle import load, dump
import shutil
import time
from numpy import *
import csv,errno
from scipy.stats import mode, kurtosis, skew
from ChalearnLAPEvaluation import evalGesture
from itertools import tee, islice
from multiprocessing import Process

# data = "/media/lio/64EE5F7C8CC54BFB/chalearn2014/valid"
datadir = "/media/Data/mp/chalearn2014/"
vfile = "Sample0471_color.mp4"
lbldir = "/media/Data/mp/chalearn2014/labels/"
preddir = "/home/lio/Dropbox/MP/chalearn2014/evaluation/results/predsperframe/"
dst = "/home/lio/Dropbox/MP/chalearn2014/evaluation/results/step1alltest/"
playvid=True

files = glob(preddir+"*")
files.sort()
# print len(files)
def make_sure_path_exists(path):
	try: os.makedirs(path)
	except OSError as exception:
		if exception.errno != errno.EEXIST: raise

make_sure_path_exists(dst)

def job_func(files_):
	for predfile in files_:
		proc_file(predfile)


def main():
	# def partition(iterable,parts):
	#     return [list(islice(it, i, None, parts)) for i, it in enumerate(tee(iterable, parts))]

	# # samples = partition(samples,n_div)[p_i]
	# n_p = 1
	# jobs = []
	# parts = partition(files,n_p)
	# for i in range(n_p):
	# 	jobs.append(Process(target=job_func,args=(parts[i],)))

	# for job in jobs: 
	# 	job.daemon = True
	# 	job.start()
	# for job in jobs: job.join()


	job_func(files)

	cv2.destroyAllWindows()

def predict(ps,j):
	begin = j-12
	end = j+12
	if begin < 0 : begin=0
	if begin >= ps.shape[0] : begin=ps.shape[0]-1
	if end > ps.shape[0] : end=ps.shape[0]
	if end <=0 : end=1
	b = empty(end-begin,uint8)
	# probs = empty(end-begin,float32)
	for i,p in enumerate(ps[begin:end]):
		ind = p.argmax()
		# probs[i] = p[ind]
		b[i] = ind+1

	# print array(b,uint8)
	m = int(mode(b)[0][0])
	# inds = where(b==m)

	if j >= ps.shape[0] : j=ps.shape[0]-1
	arg2 = ps[j][m-1]
	if m==21: m=0
	# if pr[2] > 0.5: m=0
	return m,arg2#ps[j].max()

def proc_file(predfile):
	file = predfile.split('/')
	file = file[-1].replace("_prediction.zip","_color.mp4")

	labelfile = lbldir + file.replace("_color.mp4","_labels.csv")
	datafile = lbldir + file.replace("_color.mp4","_data.csv")
	predfile = preddir + file.replace("_color.mp4","_prediction.zip")

	# print labelfile
	print file
	# print predfile

	n_fr = 0
	with GzipFile(predfile,'rb') as f:
		ps = load(f)
		n_fr = ps.shape[0]+32

	predstr = empty((n_fr,3),dtype="object")
	pred = empty((n_fr,3),dtype=float32)

	predlbl = empty((n_fr),dtype=uint8)

	# print n_fr

	with GzipFile(predfile,'rb') as f:
		ps = load(f)

		#extend ps
		ps_new = empty((n_fr,21),dtype=float32)
		ps_new[:ps.shape[0]] = ps
		ps_new[ps.shape[0]:] = ps[-1:].repeat(n_fr-ps.shape[0],axis=0)
		ps = ps_new
		n = ps.shape[0]
		for i in xrange(n_fr):
			# if playvid:
			# 	ind = i
			# 	if ind >= n: ind = n-1
			# 	p = ps[ind]
			# 	l = list(p)
			# 	l.sort(reverse=True)
			# 	top3 = l[:3]
			# 	# print top3

			# 	predstr[i,0] = str(where(p==top3[0])[0][0]+1) + " "+str(int(round(top3[0]*1000)) /1000.)[:4]
			# 	# pred[i,0] = top3[0]
			# 	# pred[i,1] = where(p==top3[1])[0][0]+1
			# 	# pred[i,2] = top3[1]
			# 	predstr[i,1] = str(where(p==top3[1])[0][0]+1) + " "+str(int(round(top3[1]*1000)) /1000.)[:4]
			# 	predstr[i,2] = str(where(p==top3[2])[0][0]+1) + " "+str(int(round(top3[2]*1000)) /1000.)[:4]

			# b = nb(ps,i)

			predlbl[i], pred[i,0] = predict(ps,i)

	# ps_new = empty((n_fr,20),dtype=float32)
	# for i in xrange(n_fr):
	# 	m = predlbl[i] -1 
	# 	if m == -1: m = 20
	# 	ps_new[i,:m] = ps[i,:m]
	# 	ps_new[i,m:] = ps[i,m+1:]
	# ps = ps_new

	prev = -1
	csvpred = []
	csvbeg = []
	csvend = []
	count, bi = 0,0
	probs, probs2 = [], []
	for i,p in enumerate(predlbl):
		if prev != p:
			# probs.append(pred[i,0])
			# probsa = array(probs)
			# probsb = array(probs2)
			# print probsa.max(), probsa.mean() 
			detected = False
			falsepos = False
			if prev>0 and pred[bi:i,0].max()>0.999 \
				and count >= 0: 
				detected = True 

			# if i-bi > 1:
			# 	b = ps[bi:i].copy()
			# 	b = b.argsort()
			# 	b = where(b==20)[1]
			# 	# print b

			# if prev>0 and pred[bi:i,0].max()>0.99 and ps[bi:i,20].mean()>0.1 : falsepos = True

			if detected or (prev>0 and count > 12 and pred[bi:i,0].max()>0.89 and not falsepos):  
				# k = kurtosis(ps[bi:i])
				a = ps[bi:i,20]
				# # a.sort()
				# b = ps[bi:i]
				# b = b[:,:18]
				# s = skew(b)
				# if a.mean()*1e+3 >= 9: print "\nBINGO\n"
				# print a.mean()
				if detected or a.min() < 0.007: 
					if p>0:
						end = i+16 
					else:
						end = i+21 

						j = 1
						while i-j >=0 and ps[i-j,20] > 0.51: 
							j += 1
							end -=1
						if j==1:
							j = 1
							n = ps.shape[0]
							while i+j < n and ps[i+j,20] < 0.35: 
								j += 1
								end +=1


						

					if end > n_fr: end = n_fr
					if count <12:
						begin -= 2

					csvpred.append(prev)
					csvbeg.append(begin)
					csvend.append(end)

			
			count = 0
			# probs = []
			if p>0:

				bi = i
				if i<8:
					begin = i + 7 
				else:
					begin = i+12
				if begin < 1: begin = 1

				
				if prev>0:
					begin +=2

					j = 1
					while i-j >=0 and ps[i-j,p-1] > 0.81: 
						j += 1
						begin -=1
					if j==1:
						j = 1
						n = ps.shape[0]
						while i+j < n and ps[i+j,p-1] < 0.29: 
							j += 1
							begin +=1

				else:
					j = 1
					while i-j >=0 and ps[i-j,p-1] > 0.87: 
						j += 1
						begin -=1
					if j==1:
						j = 1
						n = ps.shape[0]
						while i+j < n and ps[i+j,p-1] < 0.28: 
							j += 1
							begin +=1


				# if i>0 and ps[i-1,p-1] > 0.8:
				# 	begin -=1


			prev = p
		else:
			count += 1
			# probs.append(pred[i,0])
			# probs2.append(pred[i,1])

	if prev>0:
		csvpred.append(prev)
		csvbeg.append(begin)
		csvend.append(i)

	dstfile = dst + file.replace("_color.mp4","_prediction.csv")
	s = ""
	for i in xrange(len(csvpred)):
		s += str(csvpred[i])+","+str(csvbeg[i])+","+str(csvend[i])+"\n"
		# print s
	with open(dstfile,"w") as f: 
		f.write(s)




	if playvid and file == vfile:
	# if False:
		vsk = VideoSink('/home/lio/Desktop/video3.avi',rate=20, 
            size=(480,640),colorspace='bgr24', codec="x264")
		pred = zeros((n_fr,),"object")
		gt = zeros((n_fr,),"object")
		for i in xrange(len(csvpred)):
			pred[csvbeg[i]-1:csvend[i]-1] = csvpred[i]

		with open(labelfile, 'rb') as csvlblfile:
		    csvgt = csv.reader(csvlblfile)
		    for row in csvgt:
		        gt[int(row[1])-1:int(row[2])-1] = int(row[0])

		pred=map(label,pred)
		gt=map(label,gt)

		skip = 10
		vidfile = datadir + file
		vid = cv2.VideoCapture(vidfile)
		cur_fr = 0
		# while True:
		# 	cur_fr += 1
		for cur_fr in xrange(1,n_fr):
			res,img = vid.read()
			if res == False: break
			#print '\r{0}'.format(cur_fr)
			# sys.stdout.write('\r'+str(cur_fr).zfill(5))
			# sys.stdout.flush()

			# cv2.putText(img, "frame:"+str(cur_fr), (0,30), cv2.FONT_HERSHEY_COMPLEX , 1,(255,255,0),2)
			n1 = 30
			c = (0,0,0)
			c2 = (255,255,255)
			cv2.rectangle(img, (0,n1-30), (300,n1+40), c2,-1)#[, thickness[, lineType[, shift]]])
			cv2.putText(img, "Manueel: ", (0,n1), cv2.FONT_HERSHEY_COMPLEX , 1,c,2)
			cv2.putText(img, gt[cur_fr-1], (0,n1+30), cv2.FONT_HERSHEY_COMPLEX , 1,c,2)
			n2=30
			nx = 300
			cv2.rectangle(img, (nx,n2-30), (640,n2+40), c2,-1)#[, thickness[, lineType[, shift]]])
			cv2.putText(img, "Automatisch: ", (nx,n2), cv2.FONT_HERSHEY_COMPLEX , 1,c,2)
			cv2.putText(img, pred[cur_fr-1], (nx,n2+30), cv2.FONT_HERSHEY_COMPLEX , 1,c,2)
			# cv2.putText(img, "pred1:"+predstr[cur_fr-1,0], (0,120), cv2.FONT_HERSHEY_COMPLEX , 1,(255,255,0),2)
			# cv2.putText(img, "pred2:"+predstr[cur_fr-1,1], (0,150), cv2.FONT_HERSHEY_COMPLEX , 1,(255,255,0),2)
			# cv2.putText(img, "pred3:"+predstr[cur_fr-1,2], (0,180), cv2.FONT_HERSHEY_COMPLEX , 1,(255,255,0),2)

			vsk(img)
			# cv2.imshow("Video",img)
			# key = cv2.waitKey(50)

			# if key==27: break
			# if key==54:
			# 	vid.set(cv.CV_CAP_PROP_POS_FRAMES, cur_fr+skip-1)
			# 	cur_fr += skip-1
			# if key==52:
			# 	if cur_fr-skip-1 <0: skip = cur_fr-1
			# 	vid.set(cv.CV_CAP_PROP_POS_FRAMES, cur_fr-skip-1)
			# 	cur_fr -= skip+1
			# if key==97:
			# 	if cur_fr-2 <0: cur_fr = 2
			# 	vid.set(cv.CV_CAP_PROP_POS_FRAMES, cur_fr-2)
			# 	cur_fr -= 2
		vsk.close()
label_names = ['vattene', 'vieniqui', 'perfetto', 'furbo', 'cheduepalle', 'chevuoi', 'daccordo','seipazzo','combinato', 'freganiente', 'ok', 'cosatifarei', 'basta', 'prendere', 'noncenepiu', 'fame', 'tantotempo', 'buonissimo', 'messidaccordo', 'sonostufo']
def label(ind):
	if ind == 0: return "-"
	try:
		return label_names[ind-1]
	except: print ind

import subprocess
class VideoSink(object) :

    def __init__( self, filename='output.avi', size=(512,512), rate=30, colorspace='rgb24',codec='lavc'):
 
        # row/col --> x/y by swapping order
        self.size = size[::-1]

        cmdstring  = (  'mencoder',
            '/dev/stdin',
            '-demuxer', 'rawvideo',
            '-rawvideo', 'w=%i:h=%i'%self.size+':fps=%i:format=%s'%(rate,colorspace),
            '-o', filename,
            '-ovc', codec,
            '-nosound',
            '-really-quiet'
            )
        self.p = subprocess.Popen(cmdstring, stdin=subprocess.PIPE, shell=False)

    def __call__(self, frame) :
        assert frame.shape[0:2][::-1] == self.size
        # frame.tofile(self.p.stdin) # should be faster but it is indeed slower
        self.p.stdin.write(frame.tostring())
    def close(self) :
        self.p.stdin.close()
        self.p.terminate()

if __name__ == '__main__':
	main()