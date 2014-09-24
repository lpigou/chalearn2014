from cPickle import dump, load
import gzip
import cv2, os

def play_vid(vid, wait=50):
    import cv2
    for i,img in enumerate(vid):
        cv2.normalize(img,img, 0, 255, cv2.NORM_MINMAX)
        img = cv2.resize(img.astype("uint8"), (200,200))
        cv2.imshow("Gesture", img)
        cv2.waitKey(wait)
    cv2.destroyAllWindows()

os.system("SayStatic.exe test")

file = gzip.GzipFile("samples.zip", 'rb')
samples = load(file)
file.close()

for sample in samples:
    vid, skel = sample

    # print skel

    print "gray"
    play_vid(vid[0])
    print "depth"
    play_vid(vid[1])
    print "user"
    play_vid(vid[2])




