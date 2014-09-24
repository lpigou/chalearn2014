#-------------------------------------------------------------------------------
# Name:        Chalearn LAP sample
# Purpose:     Provide easy access to Chalearn LAP challenge data samples
#
# Author:      Xavier Baro
#
# Created:     21/01/2014
# Copyright:   (c) Xavier Baro 2014
# Licence:     <your licence>
#-------------------------------------------------------------------------------
import os
import zipfile
import shutil
import cv2
import numpy
import csv
from PIL import Image, ImageDraw


class Skeleton(object):
    """ Class that represents the skeleton information """
    #define a class to encode skeleton data
    def __init__(self,data):
        """ Constructor. Reads skeleton information from given raw data """
        # Create an object from raw data
        self.joins=dict();
        pos=0
        self.joins['HipCenter']=(map(float,data[pos:pos+3]),map(float,data[pos+3:pos+7]),map(int,data[pos+7:pos+9]))
        pos=pos+9
        self.joins['Spine']=(map(float,data[pos:pos+3]),map(float,data[pos+3:pos+7]),map(int,data[pos+7:pos+9]))
        pos=pos+9
        self.joins['ShoulderCenter']=(map(float,data[pos:pos+3]),map(float,data[pos+3:pos+7]),map(int,data[pos+7:pos+9]))
        pos=pos+9
        self.joins['Head']=(map(float,data[pos:pos+3]),map(float,data[pos+3:pos+7]),map(int,data[pos+7:pos+9]))
        pos=pos+9
        self.joins['ShoulderLeft']=(map(float,data[pos:pos+3]),map(float,data[pos+3:pos+7]),map(int,data[pos+7:pos+9]))
        pos=pos+9
        self.joins['ElbowLeft']=(map(float,data[pos:pos+3]),map(float,data[pos+3:pos+7]),map(int,data[pos+7:pos+9]))
        pos=pos+9
        self.joins['WristLeft']=(map(float,data[pos:pos+3]),map(float,data[pos+3:pos+7]),map(int,data[pos+7:pos+9]))
        pos=pos+9
        self.joins['HandLeft']=(map(float,data[pos:pos+3]),map(float,data[pos+3:pos+7]),map(int,data[pos+7:pos+9]))
        pos=pos+9
        self.joins['ShoulderRight']=(map(float,data[pos:pos+3]),map(float,data[pos+3:pos+7]),map(int,data[pos+7:pos+9]))
        pos=pos+9
        self.joins['ElbowRight']=(map(float,data[pos:pos+3]),map(float,data[pos+3:pos+7]),map(int,data[pos+7:pos+9]))
        pos=pos+9
        self.joins['WristRight']=(map(float,data[pos:pos+3]),map(float,data[pos+3:pos+7]),map(int,data[pos+7:pos+9]))
        pos=pos+9
        self.joins['HandRight']=(map(float,data[pos:pos+3]),map(float,data[pos+3:pos+7]),map(int,data[pos+7:pos+9]))
        pos=pos+9
        self.joins['HipLeft']=(map(float,data[pos:pos+3]),map(float,data[pos+3:pos+7]),map(int,data[pos+7:pos+9]))
        pos=pos+9
        self.joins['KneeLeft']=(map(float,data[pos:pos+3]),map(float,data[pos+3:pos+7]),map(int,data[pos+7:pos+9]))
        pos=pos+9
        self.joins['AnkleLeft']=(map(float,data[pos:pos+3]),map(float,data[pos+3:pos+7]),map(int,data[pos+7:pos+9]))
        pos=pos+9
        self.joins['FootLeft']=(map(float,data[pos:pos+3]),map(float,data[pos+3:pos+7]),map(int,data[pos+7:pos+9]))
        pos=pos+9
        self.joins['HipRight']=(map(float,data[pos:pos+3]),map(float,data[pos+3:pos+7]),map(int,data[pos+7:pos+9]))
        pos=pos+9
        self.joins['KneeRight']=(map(float,data[pos:pos+3]),map(float,data[pos+3:pos+7]),map(int,data[pos+7:pos+9]))
        pos=pos+9
        self.joins['AnkleRight']=(map(float,data[pos:pos+3]),map(float,data[pos+3:pos+7]),map(int,data[pos+7:pos+9]))
        pos=pos+9
        self.joins['FootRight']=(map(float,data[pos:pos+3]),map(float,data[pos+3:pos+7]),map(int,data[pos+7:pos+9]))
    def getAllData(self):
        """ Return a dictionary with all the information for each skeleton node """
        return self.joins
    def getWorldCoordinates(self):
        """ Get World coordinates for each skeleton node """
        skel=dict()
        for key in self.joins.keys():
            skel[key]=self.joins[key][0]
        return skel
    def getJoinOrientations(self):
        """ Get orientations of all skeleton nodes """
        skel=dict()
        for key in self.joins.keys():
            skel[key]=self.joins[key][1]
        return skel
    def getPixelCoordinates(self):
        """ Get Pixel coordinates for each skeleton node """
        skel=dict()
        for key in self.joins.keys():
            skel[key]=self.joins[key][2]
        return skel
    def toImage(self,width,height,bgColor):
        """ Create an image for the skeleton information """
        SkeletonConnectionMap = (['HipCenter','Spine'],['Spine','ShoulderCenter'],['ShoulderCenter','Head'],['ShoulderCenter','ShoulderLeft'], \
                                 ['ShoulderLeft','ElbowLeft'],['ElbowLeft','WristLeft'],['WristLeft','HandLeft'],['ShoulderCenter','ShoulderRight'], \
                                 ['ShoulderRight','ElbowRight'],['ElbowRight','WristRight'],['WristRight','HandRight'],['HipCenter','HipRight'], \
                                 ['HipRight','KneeRight'],['KneeRight','AnkleRight'],['AnkleRight','FootRight'],['HipCenter','HipLeft'], \
                                 ['HipLeft','KneeLeft'],['KneeLeft','AnkleLeft'],['AnkleLeft','FootLeft'])
        im = Image.new('RGB', (width, height), bgColor)
        draw = ImageDraw.Draw(im)
        for link in SkeletonConnectionMap:
            p=self.getPixelCoordinates()[link[1]]
            p.extend(self.getPixelCoordinates()[link[0]])
            draw.line(p, fill=(255,0,0), width=5)
        for node in self.getPixelCoordinates().keys():
            p=self.getPixelCoordinates()[node]
            r=5
            draw.ellipse((p[0]-r,p[1]-r,p[0]+r,p[1]+r),fill=(0,0,255))
        del draw
        image = numpy.array(im)
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image


class GestureSample(object):
    """ Class that allows to access all the information for a certain gesture database sample """
    #define class to access gesture data samples
    def __init__ (self,fileName):
        """ Constructor. Read the sample file and unzip it if it is necessary. All the data is loaded.

            sample=GestureSample('Sample0001.zip')

        """
        # Check the given file
        if not os.path.exists(fileName) or not os.path.isfile(fileName):
            raise Exception("Sample path does not exist: " + fileName)

        # Prepare sample information
        self.fullFile = fileName
        self.dataPath = os.path.split(fileName)[0]
        self.file=os.path.split(fileName)[1]
        self.seqID=os.path.splitext(self.file)[0]
        self.samplePath=self.dataPath + os.path.sep + self.seqID;

        # Unzip sample if it is necessary
        if os.path.isdir(self.samplePath) :
            self.unzip = False
        else:
            self.unzip = True
            zipFile=zipfile.ZipFile(self.fullFile,"r")
            zipFile.extractall(self.samplePath)

        # Open video access for RGB information
        rgbVideoPath=self.samplePath + os.path.sep + self.seqID + '_color.mp4'
        if not os.path.exists(rgbVideoPath):
            raise Exception("Invalid sample file. RGB data is not available")
        self.rgb = cv2.VideoCapture(rgbVideoPath)
        while not self.rgb.isOpened():
            self.rgb = cv2.VideoCapture(rgbVideoPath)
            cv2.waitKey(500)
            # Open video access for Depth information
        depthVideoPath=self.samplePath + os.path.sep + self.seqID + '_depth.mp4'
        if not os.path.exists(depthVideoPath):
            raise Exception("Invalid sample file. Depth data is not available")
        self.depth = cv2.VideoCapture(depthVideoPath)
        while not self.depth.isOpened():
            self.depth = cv2.VideoCapture(depthVideoPath)
            cv2.waitKey(500)
            # Open video access for User segmentation information
        userVideoPath=self.samplePath + os.path.sep + self.seqID + '_user.mp4'
        if not os.path.exists(userVideoPath):
            raise Exception("Invalid sample file. User segmentation data is not available")
        self.user = cv2.VideoCapture(userVideoPath)
        while not self.user.isOpened():
            self.user = cv2.VideoCapture(userVideoPath)
            cv2.waitKey(500)
            # Read skeleton data
        skeletonPath=self.samplePath + os.path.sep + self.seqID + '_skeleton.csv'
        if not os.path.exists(skeletonPath):
            raise Exception("Invalid sample file. Skeleton data is not available")
        self.skeletons=[]
        with open(skeletonPath, 'rb') as csvfile:
            filereader = csv.reader(csvfile, delimiter=',')
            for row in filereader:
                self.skeletons.append(Skeleton(row))
            del filereader
            # Read sample data
        sampleDataPath=self.samplePath + os.path.sep + self.seqID + '_data.csv'
        if not os.path.exists(sampleDataPath):
            raise Exception("Invalid sample file. Sample data is not available")
        self.data=dict()
        with open(sampleDataPath, 'rb') as csvfile:
            filereader = csv.reader(csvfile, delimiter=',')
            for row in filereader:
                self.data['numFrames']=int(row[0])
                self.data['fps']=int(row[1])
                self.data['maxDepth']=int(row[2])
            del filereader
            # Read labels data
        labelsPath=self.samplePath + os.path.sep + self.seqID + '_labels.csv'
        if os.path.exists(labelsPath):
            # warnings.warn("Labels are not available", Warning)
            self.labels=[]
            with open(labelsPath, 'rb') as csvfile:
                filereader = csv.reader(csvfile, delimiter=',')
                for row in filereader:
                    self.labels.append(map(int,row))
                del filereader
            # print self.labels
    def __del__(self):
        """ Destructor. If the object unziped the sample, it remove the temporal data """
        if self.unzip:
            self.clean()
    def clean(self):
        """ Clean temporal unziped data """
        try:
            del self.rgb;
            del self.depth;
            del self.user;
            shutil.rmtree(self.samplePath)
        except: pass
    def getFrame(self,video, frameNum):
        """ Get a single frame from given video object """
        # Check frame number
        # Get total number of frames
        numFrames = video.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
        # Check the given file
        if frameNum<1 or frameNum>numFrames:
            raise Exception("Invalid frame number <" + str(frameNum) + ">. Valid frames are values between 1 and " + str(int(numFrames)))
            # Set the frame index
        video.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,frameNum-1)
        ret,frame=video.read()
        if ret==False:
            raise Exception("Cannot read the frame")
        return frame
    def getRGB(self, frameNum):
        """ Get the RGB color image for the given frame """
        #get RGB frame
        return self.getFrame(self.rgb,frameNum)
    def getGray(self, frameNum):
        gray = self.getFrame(self.rgb,frameNum)
        gray=cv2.cvtColor(gray,cv2.cv.CV_RGB2GRAY)
        return gray
    def getDepth(self, frameNum):
        """ Get the depth image for the given frame """
        #get Depth frame
        depthData=self.getFrame(self.depth,frameNum)
        # Convert to grayscale
        depth=cv2.cvtColor(depthData,cv2.cv.CV_RGB2GRAY)
        # Convert to float point
        # depth=depthGray.astype(numpy.float32)
        # Convert to depth values
        # depth=depth/255.0*float(self.data['maxDepth'])
        # depth=depth.round()
        # depth=depth.astype(numpy.uint16)
        return depth
    def getUser(self, frameNum):
        """ Get user segmentation image for the given frame """
        #get user segmentation frame
        user= self.getFrame(self.user,frameNum)
        user=cv2.cvtColor(user,cv2.cv.CV_RGB2GRAY)
        user[user<128] = 0
        user[user>128] = 1
        return user
    def getSkeleton(self, frameNum):
        """ Get the skeleton information for a given frame. It returns a Skeleton object """
        #get user skeleton for a given frame
        # Check frame number
        # Get total number of frames
        numFrames = len(self.skeletons)
        # Check the given file
        if frameNum<1 or frameNum>numFrames:
            raise Exception("Invalid frame number <" + str(frameNum) + ">. Valid frames are values between 1 and " + str(int(numFrames)))
        return self.skeletons[frameNum-1]
    def getSkeletonImage(self, frameNum):
        """ Create an image with the skeleton image for a given frame """
        return self.getSkeleton(frameNum).toImage(640,480,(255,255,255))
    def getNumFrames(self):
        """ Get the number of frames for this sample """
        return self.data['numFrames']
    def getComposedFrame(self, frameNum):
        """ Get a composition of all the modalities for a given frame """
        # get sample modalities
        rgb=self.getRGB(frameNum)
        depthValues=self.getDepth(frameNum)
        user=self.getUser(frameNum)
        skel=self.getSkeletonImage(frameNum)

        # Build depth image
        depth = depthValues.astype(numpy.float32)
        depth = depth*255.0/float(self.data['maxDepth'])
        depth = depth.round()
        depth = depth.astype(numpy.uint8)
        depth = cv2.applyColorMap(depth,cv2.COLORMAP_JET)

        # Build final image
        compSize1=(max(rgb.shape[0],depth.shape[0]),rgb.shape[1]+depth.shape[1])
        compSize2=(max(user.shape[0],skel.shape[0]),user.shape[1]+skel.shape[1])
        comp = numpy.zeros((compSize1[0]+ compSize2[0],max(compSize1[1],compSize2[1]),3), numpy.uint8)

        # Create composition
        comp[:rgb.shape[0],:rgb.shape[1],:]=rgb
        comp[:depth.shape[0],rgb.shape[1]:rgb.shape[1]+depth.shape[1],:]=depth
        comp[compSize1[0]:compSize1[0]+user.shape[0],:user.shape[1],:]=user
        comp[compSize1[0]:compSize1[0]+skel.shape[0],user.shape[1]:user.shape[1]+skel.shape[1],:]=skel

        return comp
    def getGestures(self):
        """ Get the list of gesture for this sample. Each row is a gesture, with the format (gestureID,startFrame,endFrame) """
        return self.labels
    def getGestureName(self,gestureID):
        """ Get the gesture label from a given gesture ID """
        names=('vattene','vieniqui','perfetto','furbo','cheduepalle','chevuoi','daccordo','seipazzo', \
               'combinato','freganiente','ok','cosatifarei','basta','prendere','noncenepiu','fame','tantotempo', \
               'buonissimo','messidaccordo','sonostufo')
        # Check the given file
        if gestureID<1 or gestureID>20:
            raise Exception("Invalid gesture ID <" + str(gestureID) + ">. Valid IDs are values between 1 and 20")
        return names[gestureID-1]
    def exportPredictions(self, prediction,predPath):
        """ Export the given prediction to the correct file in the given predictions path """
        if not os.path.exists(predPath):
            os.makedirs(predPath)
        output_filename = os.path.join(predPath,  self.seqID + '_prediction.csv')
        output_file = open(output_filename, 'wb')
        for row in prediction:
            output_file.write(repr(int(row[0])) + "," + repr(int(row[1])) + "," + repr(int(row[2])) + "\n")
        output_file.close()
    def evaluate(self,csvpathpred):
        """ Evaluate this sample agains the ground truth file """
        maxGestures=11
        seqLength=self.getNumFrames()

        # Get the list of gestures from the ground truth and frame activation
        predGestures = []
        binvec_pred = numpy.zeros((maxGestures, seqLength))
        gtGestures = []
        binvec_gt = numpy.zeros((maxGestures, seqLength))
        with open(csvpathpred, 'rb') as csvfilegt:
            csvgt = csv.reader(csvfilegt)
            for row in csvgt:
                binvec_pred[int(row[0])-1, int(row[1])-1:int(row[2])-1] = 1
                predGestures.append(int(row[0]))

        # Get the list of gestures from prediction and frame activation
        for row in self.getActions():
                binvec_gt[int(row[0])-1, int(row[1])-1:int(row[2])-1] = 1
                gtGestures.append(int(row[0]))

        # Get the list of gestures without repetitions for ground truth and predicton
        gtGestures = numpy.unique(gtGestures)
        predGestures = numpy.unique(predGestures)

        # Find false positives
        falsePos=numpy.setdiff1d(gtGestures, numpy.union1d(gtGestures,predGestures))

        # Get overlaps for each gesture
        overlaps = []
        for idx in gtGestures:
            intersec = sum(binvec_gt[idx-1] * binvec_pred[idx-1])
            aux = binvec_gt[idx-1] + binvec_pred[idx-1]
            union = sum(aux > 0)
            overlaps.append(intersec/union)

        # Use real gestures and false positive gestures to calculate the final score
        return sum(overlaps)/(len(overlaps)+len(falsePos))

if __name__ == '__main__':
    # data = "/media/Data/mp/chalearn2014/train_raw"
    data = "/home/lio/mp/chalearn2014/train_raw"
    # Get the list of training samples
    samples=os.listdir(data)
    print len(samples)
    def analyse(mat):
        print mat.shape, mat.min(), mat.mean(), mat.max(), mat.std()

    for file in samples[:2]:
        if not file.endswith(".zip"): continue
        print("\t Processing file " + file)
        # Create the object to access the sample
        smp=GestureSample(os.path.join(data,file))
        # Get the list of actions for this frame
        gesturesList=smp.getGestures()
        # Iterate for each action in this sample
        for gesture in gesturesList[2:3]:
            # Get the gesture ID, and start and end frames for the gesture
            gestureID,startFrame,endFrame=gesture

            for numFrame in range(startFrame,endFrame)[5:6]:
                # Get the RGB image for this frame
                rgb=smp.getRGB(numFrame);
                analyse(rgb)
                # Get the Depth image for this frame
                depth=smp.getDepth(numFrame);
                analyse(depth)
                # Get the user segmentation image for this frame
                user=smp.getUser(numFrame);
                print user[200:220,300:320]
                # Get the Skeleton object for this frame
                skel=smp.getSkeleton(numFrame);
                # analyse(skel)
        # Remove the sample object
        del smp;


