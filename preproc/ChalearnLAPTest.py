#-------------------------------------------------------------------------------
# Name:        Chalearn LAP utils scripts
# Purpose:     Provide scripts to add labels to Chalearn LAP challenge tracks samples
#
# Author:      Xavier Baro
#
# Created:     25/04/2014
# Copyright:   (c) Chalearn LAP 2014
# Licence:     GPL
#-------------------------------------------------------------------------------
import os
import zipfile
import shutil
import glob

def main():
    """ Main script. Created a labeled copy of validation samples """
    # Data folder (Unlabeled data samples)
    data='./data/';
    # Labels file (Unziped validation.zip)
    labels='./labels/';    
    # Output folder
    outData='./labeledData/'
    
    # Use the method for desired track
    print('Uncoment the line for your track')
    #addLabels_Track1(data, labels, outData)
    #addLabels_Track2(data, labels, outData)
    #addLabels_Track3(data, labels, outData)
    

def addLabels_Track1(dataPath, labelsPath, outputPath):
    """ Add labels to the samples"""
    
    # Check the given data path
    if not os.path.exists(dataPath) or not os.path.isdir(dataPath):
        raise Exception("Data path does not exist: " + dataPath)
    
    # Check the given labels path
    if not os.path.exists(labelsPath) or not os.path.isdir(labelsPath):
	raise Exception("Labels path does not exist: " + labelsPath)    

    # Check the output path
    if os.path.exists(outputPath) and os.path.isdir(outputPath):
        raise Exception("Output path already exists. Remove it before start: " + outputPath)

    # Create the output path
    os.makedirs(outputPath)
    if not os.path.exists(outputPath) or not os.path.isdir(outputPath):
        raise Exception("Cannot create the output path: " + outputPath)

    # Get the list of samples
    samplesList = os.listdir(dataPath)

    # For each sample on the GT, search the given prediction
    for sample in samplesList:
        # Build paths for sample
    	sampleFile = os.path.join(dataPath, sample)

        # Check that is a ZIP file
        if not os.path.isfile(sampleFile) or not sample.lower().endswith(".zip"):
            continue

        # Prepare sample information
        file = os.path.split(sampleFile)[1]
        sampleID = os.path.splitext(file)[0]
        samplePath = dataPath + os.path.sep + sampleID

        # Unzip sample if it is necessary
        if os.path.isdir(samplePath):
            unziped = False
        else:
            unziped = True
            zipFile = zipfile.ZipFile(sampleFile, "r")
            zipFile.extractall(samplePath)

	# Create the output file
	labSampleFile=zipfile.ZipFile(os.path.join(outputPath, sample), "w")
	
	# Copy the data files
	imgPath=os.path.join(samplePath,'imagesjpg')
	for img in os.listdir(imgPath):	    
	    srcSampleDataPath = os.path.join(imgPath, img)	
	    labSampleFile.write(srcSampleDataPath,os.path.join('imagesjpg',os.path.basename(srcSampleDataPath)), zipfile.ZIP_DEFLATED)
	
	# Add the labels
	seqNum=sampleID[3:len(sampleID)]
	for img in glob.glob(os.path.join(labelsPath,seqNum + '_*')):	    
	    labSampleFile.write(img,os.path.join('maskspng',os.path.basename(img)), zipfile.ZIP_DEFLATED)	
	
	# Close the output file
	labSampleFile.close()
	
        # Remove temporal data
        if unziped:
            shutil.rmtree(samplePath)

def addLabels_Track2(dataPath, labelsPath, outputPath):
    """ Add labels to the samples"""
    
    # Check the given data path
    if not os.path.exists(dataPath) or not os.path.isdir(dataPath):
        raise Exception("Data path does not exist: " + dataPath)
    
    # Check the given labels path
    if not os.path.exists(labelsPath) or not os.path.isdir(labelsPath):
	raise Exception("Labels path does not exist: " + labelsPath)    

    # Check the output path
    if os.path.exists(outputPath) and os.path.isdir(outputPath):
        raise Exception("Output path already exists. Remove it before start: " + outputPath)

    # Create the output path
    os.makedirs(outputPath)
    if not os.path.exists(outputPath) or not os.path.isdir(outputPath):
        raise Exception("Cannot create the output path: " + outputPath)

    # Get the list of samples
    samplesList = os.listdir(dataPath)

    # For each sample on the GT, search the given prediction
    for sample in samplesList:
        # Build paths for sample
    	sampleFile = os.path.join(dataPath, sample)

        # Check that is a ZIP file
        if not os.path.isfile(sampleFile) or not sample.lower().endswith(".zip"):
            continue

        # Prepare sample information
        file = os.path.split(sampleFile)[1]
        sampleID = os.path.splitext(file)[0]
        samplePath = dataPath + os.path.sep + sampleID

        # Unzip sample if it is necessary
        if os.path.isdir(samplePath):
            unziped = False
        else:
            unziped = True
            zipFile = zipfile.ZipFile(sampleFile, "r")
            zipFile.extractall(samplePath)

	# Create the output file
	labSampleFile=zipfile.ZipFile(os.path.join(outputPath, sample), "w")
	
	# Copy the data files
	for data in ('_color.mp4', '_data.csv'):
	    srtFileName=sampleID + data
	    srcSampleDataPath = os.path.join(samplePath, srtFileName)
	    if not os.path.exists(srcSampleDataPath) or not os.path.isfile(srcSampleDataPath):
		raise Exception("Invalid sample file. File " + strFileName + " is not available")	
	    labSampleFile.write(srcSampleDataPath,os.path.basename(srcSampleDataPath), zipfile.ZIP_DEFLATED)
	
	# Add the labels
	srtFileName=sampleID + '_labels.csv'
	srcSampleDataPath = os.path.join(labelsPath, srtFileName)
	if not os.path.exists(srcSampleDataPath) or not os.path.isfile(srcSampleDataPath):
	    raise Exception("Invalid sample file. Labels file is not available")	
	labSampleFile.write(srcSampleDataPath,os.path.basename(srcSampleDataPath), zipfile.ZIP_DEFLATED)	
	
	# Close the output file
	labSampleFile.close()
	
        # Remove temporal data
        if unziped:
            shutil.rmtree(samplePath)

def addLabels_Track3(dataPath, labelsPath, outputPath):
    """ Add labels to the samples"""
    
    # Check the given data path
    if not os.path.exists(dataPath) or not os.path.isdir(dataPath):
        raise Exception("Data path does not exist: " + dataPath)
    
    # Check the given labels path
    if not os.path.exists(labelsPath) or not os.path.isdir(labelsPath):
	raise Exception("Labels path does not exist: " + labelsPath)    

    # Check the output path
    if os.path.exists(outputPath) and os.path.isdir(outputPath):
        raise Exception("Output path already exists. Remove it before start: " + outputPath)

    # Create the output path
    os.makedirs(outputPath)
    if not os.path.exists(outputPath) or not os.path.isdir(outputPath):
        raise Exception("Cannot create the output path: " + outputPath)

    # Get the list of samples
    samplesList = os.listdir(dataPath)

    # For each sample on the GT, search the given prediction
    for sample in samplesList:
        # Build paths for sample
    	sampleFile = os.path.join(dataPath, sample)

        # Check that is a ZIP file
        if not os.path.isfile(sampleFile) or not sample.lower().endswith(".zip"):
            continue

        # Prepare sample information
        file = os.path.split(sampleFile)[1]
        sampleID = os.path.splitext(file)[0]
        samplePath = dataPath + os.path.sep + sampleID

        # Unzip sample if it is necessary
        if os.path.isdir(samplePath):
            unziped = False
        else:
            unziped = True
            zipFile = zipfile.ZipFile(sampleFile, "r")
            zipFile.extractall(samplePath)

	# Create the output file
	labSampleFile=zipfile.ZipFile(os.path.join(outputPath, sample), "w")
	
	# Copy the data files
	for data in ('_color.mp4', '_depth.mp4', '_user.mp4', '_skeleton.csv', '_data.csv'):
	    srtFileName=sampleID + data
	    srcSampleDataPath = os.path.join(samplePath, srtFileName)
	    if not os.path.exists(srcSampleDataPath) or not os.path.isfile(srcSampleDataPath):
		raise Exception("Invalid sample file. File " + strFileName + " is not available")	
	    labSampleFile.write(srcSampleDataPath,os.path.basename(srcSampleDataPath), zipfile.ZIP_DEFLATED)
	
	# Add the labels
	srtFileName=sampleID + '_labels.csv'
	srcSampleDataPath = os.path.join(labelsPath, srtFileName)
	if not os.path.exists(srcSampleDataPath) or not os.path.isfile(srcSampleDataPath):
	    raise Exception("Invalid sample file. Labels file is not available")	
	labSampleFile.write(srcSampleDataPath,os.path.basename(srcSampleDataPath), zipfile.ZIP_DEFLATED)	
	
	# Close the output file
	labSampleFile.close()
	
        # Remove temporal data
        if unziped:
            shutil.rmtree(samplePath)

if __name__ == '__main__':
    main()
