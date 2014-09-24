**********************************
****  Chalearn Looking at People 2014  ****
**********************************

This ZIP file contains the code provided for the Chalearn LAP competition:

	- ChalearnLAPEvaluation.py: Contains the methods for evaluation purposes.
	- ChalearnLAPSample.py: Contains the objects to access the provided data samples.
	- data: This folder contains some training samples.
	- track3_demo.py: Main file, showing how to access the data, evaluate and create the final submission file.
	
Each of the methods used in this scripts are described on the Codalab.org competition, and in the code section for each track in the downloads page (sunai.uoc.edu/chalearnLAP/).

In order to use this code, you need Python 2.7 and the following libraries:

	- NumPy: Fundamental package for scientific computation. (http://www.numpy.org/) 
	- Python Image Library (PIL): You can download it from http://www.pythonware.com/products/pil/ or use the Pillow library (https://pypi.python.org/pypi/Pillow/)
	
To run the sample, just run the fle track3_demo.py:

	python track3_demo.py
	
It will do the following steps:

	1.- Divide the samples into Training and Test sets
	2.- Learn a model from Training samples. In this case random predictions are used, therefore the model is a dummy model.
	3.- Use the model to make predictions on the Test samples.
	4.- Export the Test samples ground truth to simulate CodaLab evaluation procedure.
	5.- Evaluate the predictions of step 3 using the ground truth of step 4.
	6.- Create a submission file ready to submit to CodaLab.
	
All the code is commented in order to provide an example on how to access the data to define your own models.