"""
Tool to help design a convolutional neural network
"""

"""
filter_shapes = [[1,5,5], [1,5,5], [1,5,5]]
pool_sizes = [[4,3,3], [4,2,2], [2,4,4]]

filter_shapes = [[1,7,7], [1,6,6], [1,7,7]]
pool_sizes = [[4,2,2], [4,3,3], [2,2,2]]

filter_shapes = [[1,7,7], [1,5,5], [1,3,3]]
pool_sizes = [[4,2,2], [4,5,5], [2,3,3]]

filter_shapes = [[1,3,3], [1,5,5], [1,7,7]]
pool_sizes = [[4,2,2], [4,3,3], [2,3,3]]


filter_shapes = [[1,7,7], [5,5,5], [3,3,3]]
pool_sizes = [[4,2,2], [1,5,5], [2,3,3]]
"""

# filter_shapes = [[1,3,3], [1,15,15], [3,6,6]]
# pool_sizes = [[2,2,2], [2,2,2], [2,2,2]]
filter_shapes = [[1,4,4], [1,4,4],[1,4,4]]
pool_sizes = [[2,1,1], [2,2,2],[2,2,2]]


# vid_res = (32,90,90)
# vid_res = (32,64,64)
vid_res = (32,32,32)
init_maps = 4
n_layers = 3
nk = [20,100,16*25*2,50,50]
batch_size = 5


fix_time = True
same_size = True


nk = [init_maps]+nk
ims = [batch_size, init_maps]+list(vid_res)
n = len(ims)

mem = 0
for i in range(n_layers):

	fs = [nk[i+1], nk[i]]+filter_shapes[i]
	ps = pool_sizes[i]

	fr = 	[ims[n-3]-fs[n-3]+1, \
			ims[n-2]-fs[n-2]+1, \
			ims[n-1]-fs[n-1]+1]

	pr =	[1.*fr[0]/ps[0], \
			1.*fr[1]/ps[1], \
			1.*fr[2]/ps[2]]

	os = [batch_size, nk[i+1]] + pr

	pos_filt = []
	for j in range(int(ims[n-3])):
		if j==0 or j== int(ims[n-3]): continue
		if fix_time: j = filter_shapes[i][0]
		for k in range(int(ims[n-2])):
			if k==0 or k==int(ims[n-2]): continue
			for l in range(int(ims[n-1])):
				if same_size and k != l: continue
				if l==0 or l==int(ims[n-1]): continue
				if fix_time: red0 = 1.
				else: red0 = 1.*(ims[n-3]-j+1)/ps[0]
				red1 = 1.*(ims[n-2]-k+1)/ps[1]
				red2 = 1.*(ims[n-1]-l+1)/ps[2]
				if red0.is_integer() and  red1.is_integer() and red2.is_integer():
					pos_filt.append([j,k,l])

		if fix_time: break

	pos_filt = pos_filt[:50]

	print "layer", i
	print "kernel: %dx%dx%d" % (fs[n-3], fs[n-2], fs[n-1])
	print "%dx%dx%dx%d --C--> %dx%dx%dx%d --MP--> %dx%dx%dx%d" % \
		(	ims[n-4], ims[n-3], ims[n-2], ims[n-1],\
			os[n-4], fr[0], fr[1], fr[2],\
			os[n-4], int(pr[0]), int(pr[1]), int(pr[2]))
	print "--------------------------------------------------"
	print "image_shape", ims
	print "filter_shape", fs
	print "pool_size", ps
	print "filter_reduction:", fr
	print "pool_reduction:", pr
	print "Output_shape", os
	print "possible filters", pos_filt
	print "n_outputs", os[n-4]*int(pr[0])*int(pr[1])*int(pr[2])
	print "memory:", (os[n-4]* fr[0]* fr[1]* fr[2])*32./8./(2.*1024.), "MB"
	ims = os
	mem += (os[n-4]* fr[0]* fr[1]* fr[2])*32./8./(2.*1024.)

	print "\n"

print mem
