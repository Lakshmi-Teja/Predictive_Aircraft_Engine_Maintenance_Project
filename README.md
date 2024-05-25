# LSTM Networks for Predictive Aircraft Engine Maintenance 


## Problem Description
In this example I build an LSTM network in order to predict remaining useful life (or time to failure) of aircraft engines.The network uses simulated aircraft sensor values to predict when an aircraft engine will fail in the future, so that maintenance can be planned in advance.This project involves in using sensor data, historical performance records, and operational parameters to predict potential engine issues specifically predicting engine failures and forecasting Remaining Useful Life of the engines before they lead to failures or disruptions
The question to ask is "Given these aircraft engine operation and failure events history, can we predict when an in-service engine will fail?"
We re-formulate this question into two closely relevant questions and answer them using two different types of machine learning models:

	* Regression models: How many more cycles an in-service engine will last before it fails?
	* Binary classification: Is this engine going to fail within w1 cycles?

## Software Environment
* Python 3.6
* numpy 1.13.3
* matplotlib 2.0.2
* scikit-learn 0.19.0
* pandas 0.20.3
* TensorFlow 1.3.0
* Keras 2.1.1

## Data Summary
In the **Dataset** directory there are the training, test and ground truth datasets.
The training data consists of **multiple multivariate time series** with "cycle" as the time unit, together with 21 sensor readings for each cycle.
Each time series can be assumed as being generated from a different engine of the same type.
The testing data has the same data schema as the training data.
The only difference is that the data does not indicate when the failure occurs.
Finally, the ground truth data provides the number of remaining working cycles for the engines in the testing data.
The following picture shows a sample of the data: 
<p align="center">
  <img src="https://github.com/umbertogriffo/Predictive-Maintenance-using-LSTM/blob/master/Output/datasetSample.png"/>
</p>


## Extensions
We can also create a model to determine if the failure will occur in different time windows, for example, fails in the window (1,w0) or fails in the window (w0+1, w1) days, and so on. This will then be a multi-classification problem, and data will need to be preprocessed accordingly. 


## References

- [1] Deep Learning for Predictive Maintenance https://github.com/Azure/lstms_for_predictive_maintenance/blob/master/Deep%20Learning%20Basics%20for%20Predictive%20Maintenance.ipynb
- [2] Predictive Maintenance: Step 2A of 3, train and evaluate regression models https://gallery.azure.ai/Experiment/Predictive-Maintenance-Step-2A-of-3-train-and-evaluate-regression-models-2
- [3] A. Saxena and K. Goebel (2008). "Turbofan Engine Degradation Simulation Data Set", NASA Ames Prognostics Data Repository (https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/#turbofan), NASA Ames Research Center, Moffett Field, CA 
- [4] Understanding LSTM Networks http://colah.github.io/posts/2015-08-Understanding-LSTMs/
