#-------------------------------------------------------------------------------
# Name:        Starting Kit for ChaLearn LAP 2014 Track3
# Purpose:     Show basic functionality of provided code
#
# Author:      Xavier Baro
#
# Created:     19/02/2014
# Copyright:   (c) Chalearn LAP 2014
# Licence:     GPL3
#-------------------------------------------------------------------------------
import sys, os, os.path,random,numpy,zipfile
from shutil import copyfile

from ChalearnLAPEvaluation import evalGesture,exportGT_Gesture
# from ChalearnLAPSample import GestureSample

def main():
    """ Main script. Show how to perform all competition steps """
    # Data folder (Training data)
    data='./data/';
    # Train folder (output)
    outTrain='./training/train/'
    # Test folder (output)
    outTest='./training/test/'
    # Predictions folder (output)
    outPred='./results/step1all/';
    # Ground truth folder (output)
    outGT='./GT/';
    # Submision folder (output)
    outSubmision='./submision/'

    # Divide data into train and test
    # createDataSets(data,outTrain,outTest,0.3);

    # Learn your model
    # if os.path.exists("model.npy"):
    #     model=numpy.load("model.npy");
    # else:
    #     model=learnModel(outTrain);
    #     numpy.save("model",model);

    # # Predict over test dataset
    # predict(model,outTest,outPred);

    # # Create evaluation gt from labeled data
    # exportGT_Gesture(outTest,outGT);

    # Evaluate your predictions
    score=evalGesture(outPred, outGT);
    print("The score for this prediction is " + "{:.12f}".format(score));

    # Prepare submision file (only for validation and final evaluation data sets)
    # createSubmisionFile(outPred,outSubmision);

def createDataSets(dataPath,trainPath,testPath,testPer):
    """ Divide input samples into Train and Test sets """
    # Get the data files
    fileList = os.listdir(dataPath);

    # Filter input files (only ZIP files)
    sampleList=[];
    for file in fileList:
        if file.endswith(".zip"):
            sampleList.append(file);

    # Calculate the number of samples for each data set
    numSamples=len(sampleList);
    numTest=round(numSamples*testPer);
    numTrain=numSamples-numTest;

    # Create a random permutation of the samples
    random.shuffle(sampleList);

    # Create the output partitions
    if os.path.exists(trainPath):
        trainFileList = os.listdir(trainPath);
        for file in trainFileList:
            os.remove(os.path.join(trainPath,file));
    else:
        os.makedirs(trainPath);

    # Create the output partitions
    if os.path.exists(testPath):
        testFileList = os.listdir(testPath);
        for file in testFileList:
            os.remove(os.path.join(testPath,file));
    else:
        os.makedirs(testPath);

    # Copy the files
    count=0;
    for file in sampleList:
        if count<numTrain:
            copyfile(os.path.join(dataPath,file), os.path.join(trainPath,file));
        else:
            copyfile(os.path.join(dataPath,file), os.path.join(testPath,file));
        count=count+1;

def learnModel(data):
    """ Access the sample information to learn a model. """
    print("Learning the model");
    # Get the list of training samples
    samples=os.listdir(data);

    # Initialize the model
    model=[];

    # Access to each sample
    for file in samples:
        if not file.endswith(".zip"):
            continue;
        print("\t Processing file " + file)

        # Create the object to access the sample
        smp=GestureSample(os.path.join(data,file));

        # ###############################################
        # USE Ground Truth information to learn the model
        # ###############################################

        # Get the list of actions for this frame
        gesturesList=smp.getGestures();

        # Iterate for each action in this sample
        for gesture in gesturesList:
            # Get the gesture ID, and start and end frames for the gesture
            gestureID,startFrame,endFrame=gesture;

            # NOTE: We use random predictions on this example, therefore, nothing is done with the image. No model is learnt.
            # Iterate frame by frame to get the information to learn the model
            for numFrame in range(startFrame,endFrame):
                # Get the RGB image for this frame
                rgb=smp.getRGB(numFrame);
                # Get the Depth image for this frame
                depth=smp.getDepth(numFrame);
                # Get the user segmentation image for this frame
                user=smp.getUser(numFrame);
                # Get the Skeleton object for this frame
                skel=smp.getSkeleton(numFrame);



        # ###############################################

        # Remove the sample object
        del smp;

    # Return the model
    return model;

def predict(model,data,output):
    """ Access the sample information to predict the pose. """

    # Get the list of training samples
    samples=os.listdir(data);

    # Access to each sample
    for file in samples:
        # Create the object to access the sample
        smp=GestureSample(os.path.join(data,file));

        # Create a random set of actions for this sample
        numFrame=0;
        pred=[];
        while numFrame<smp.getNumFrames():
            # Generate an initial displacement
            start=numFrame+random.randint(1,100);

            # Generate the gesture duration
            end=min(start+random.randint(10,100),smp.getNumFrames());

            # Generate the action ID
            gestureID=random.randint(1,20);

            # Check if the number of frames are correct
            if start<end-1 and end<smp.getNumFrames():
                # Store the prediction
                pred.append([gestureID,start,end])

            # Move ahead
            numFrame=end+1;

        # Store the prediction
        smp.exportPredictions(pred,output);

        # Remove the sample object
        del smp;

def createSubmisionFile(predictionsPath,submisionPath):
    """ Create the submission file, ready to be submited to Codalab. """

    # Create the output path and remove any old file
    if os.path.exists(submisionPath):
        oldFileList = os.listdir(submisionPath);
        for file in oldFileList:
            os.remove(os.path.join(submisionPath,file));
    else:
        os.makedirs(submisionPath);

    # Create a ZIP with all files in the predictions path
    zipf = zipfile.ZipFile(os.path.join(submisionPath,'Submission.zip'), 'w');
    for root, dirs, files in os.walk(predictionsPath):
        for file in files:
            zipf.write(os.path.join(root, file), file, zipfile.ZIP_DEFLATED);
    zipf.close()


if __name__ == '__main__':
    main()
