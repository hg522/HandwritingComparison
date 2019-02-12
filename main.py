# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 21:08:22 2018

@author: Himanshu Garg
UBID : 50292195
"""


from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import csv
import math
from matplotlib import pyplot as plt
import time
import random
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.callbacks import EarlyStopping, TensorBoard
import sklearn.metrics as skm

startT = time.time()

HODsamePairPath = 'HumanObserved-Features-Data/same_pairs.csv'
HODfeaturePath = 'HumanObserved-Features-Data/HumanObserved-Features-Data.csv'
HODdiffPairPath = 'HumanObserved-Features-Data/diffn_pairs.csv'

GSCsamePairPath = 'GSC-Features-Data/same_pairs.csv'
GSCfeaturePath = 'GSC-Features-Data/GSC-Features.csv'
GSCdiffPairPath = 'GSC-Features-Data/diffn_pairs.csv'
TrainingPercent = 80
ValidationPercent = 10
TestPercent = 10
PHI = []


def getCatAndSubRawAndTargetData(samePairPath,featurePath,diffPairPath,setType):    
     featureDict = {}
     catdatamatrix = []
     subdatamatrix = []
     featureLen = 0
     '''
     the feature file is read here line by line and a dictionary created
     with key as the image sample name and value as the corresponding features
     '''
     with open(featurePath, 'r') as fi:
         reader = csv.reader(fi)
         for indrow,row in enumerate(reader):
             if indrow == 0:
                 continue
             if setType == 'HOD':
                 featureLen = len(row) - 2
                 featureDict[row[1]] = [int(x) for x in row[2:len(row)]]  
             else :
                 featureLen = len(row) - 1
                 featureDict[row[0]] = [int(x) for x in row[1:len(row)]]
     
     '''
     The same pair file is read into a dataframe and then converted into a
     numpy array. full data is selected in this case.
     '''
     samepairdf =  pd.read_csv(samePairPath)
     samepairArr = samepairdf.values
     s_size = len(samepairArr)
     samepairdf = []
     
     '''
     The different pair file is read into a dataframe and then converted into a
     numpy array. the same amount of data is selected equal to the same pair and 
     is picked randomly for further processing
     '''
     diffpairdf =  pd.read_csv(diffPairPath)    
     diffpairArr = diffpairdf.values
     d_size = s_size
     diffpairdf = []
     
     #samepairArr = samepairArr[random.sample(range(0, len(samepairArr)), s_size)]
     diffpairArr = diffpairArr[random.sample(range(0, len(diffpairArr)), d_size)]
     
     '''
     the same pair data and the different pair data are concatenated to form 
     one array which is iterated over to get the concatenated features and the
     subtracted features.
     '''
     mat = np.concatenate((samepairArr,diffpairArr))
     diffpairArr = []
     catdatamatrix = np.empty([len(mat),2*featureLen + 1])
     subdatamatrix = np.empty([len(mat),featureLen + 1])
     for indrow,row in enumerate(mat):
           f1 = featureDict[row[0]]
           f2 = featureDict[row[1]]
           f = f1+f2
           t = [int(row[2])]
           r = f+t
           catdatamatrix[indrow] = r
           f = [abs(x - y) for x, y in zip(f1, f2)]
           r = f + t
           subdatamatrix[indrow] = r
     mat = []
     
     '''
     The concatenated and subtracted feature matrices generated are randomly 
     shuffled so that the training doesn't become biased and also so that no all
     samples of different pairs are present in the test set that is later created.
     Same for the same pair samples.
     '''
     np.random.shuffle(catdatamatrix)
     np.random.shuffle(subdatamatrix)
     
     '''
     the zero columns are removed from the two matrices to resolve the problem 
     of matrix inversion that can occur when calculating the design matrix.
     '''
     catdatamatrix = np.delete(catdatamatrix,np.where(~catdatamatrix.any(axis=0))[0],axis = 1)
     subdatamatrix = np.delete(subdatamatrix,np.where(~subdatamatrix.any(axis=0))[0],axis = 1)
     
     '''
     the matrices are decomposed into the the data and target matrices and returned
     '''
     return catdatamatrix[:,0:len(catdatamatrix[0])-1],catdatamatrix[:,len(catdatamatrix[0])-1],subdatamatrix[:,0:len(subdatamatrix[0])-1],subdatamatrix[:,len(subdatamatrix[0])-1]

def getRandomIndexes(len1,len2):
    a = range(0,len1)
    b = range(0,len2)
    st = math.floor(len(b)/len(a))
    c = b[0::st]
    return list(c)
     
'''
Here 80% of the rawTargets read are taken in the training target data set
'''
def genTrainingTarget(targetData,TrainingPercent):
    TrainingLen = int(math.ceil(len(targetData)*(TrainingPercent*0.01)))
    t           = targetData[:TrainingLen]
    return t

'''
Here 80% of the rawData read is taken as the training data set
'''
def genTrainingData(data, TrainingPercent):
    #T_len = int(math.ceil(len(data)*0.01*TrainingPercent))
    T_len = int(math.ceil(len(data)*(TrainingPercent*0.01)))
    tdata = data[0:T_len,:]
    return tdata

'''
Here validation and test target sets are created as 10% each of the initial rawTarget data 
set
'''
def genValAndTestTarget(targetData, ValPercent, TrainingCount): 
    valSize = int(math.ceil(len(targetData)*ValPercent*0.01))
    V_End = TrainingCount + valSize
    t =targetData[TrainingCount+1:V_End]
    return t

'''
Here validation and test data sets are created as 10% each of the initial rawData data 
set
'''
def genValAndTestData(data, ValPercent, TrainingCount): 
    valSize = int(math.ceil(len(data)*ValPercent*0.01))
    V_End = TrainingCount + valSize
    dataMatrix = data[TrainingCount+1:V_End,:]
    return dataMatrix

'''
Since we are using gaussian radial basis function, bigSigma or the variances need 
to be calculated. It is assumed that the variance between features is 0. A diagnal 
matrix is generated to simplify matrix multiplication
'''
def GenerateBigSigma(Data, TrainingPercent):
    DataT       = np.transpose(Data)
    BigSigma    = np.zeros((len(DataT),len(DataT)))   #generate 18x18 matrix 
    
    TrainingLen = math.ceil(len(Data)*(TrainingPercent*0.01))     
    varVect     = []
    for i in range(0,len(Data[0])):    
        vct = []
        for j in range(0,int(TrainingLen)):     
            vct.append(DataT[i][j])    
        varVect.append(np.var(vct))             
    
    for j in range(len(DataT)): 
        BigSigma[j][j] = varVect[j] + 0.001
    BigSigma = np.dot(200,BigSigma)
   
    return BigSigma

''' here we are calculating the gaussian radial basis function matrix
of dimension : N x M, where N is the number of samples and M is the number of clusters.
We have the mu matrix containing center points for M clusters.
we generate Phi matrix by iterating over the whole data by number of clusters  and 
calculating the radial basis function for each row taking the Mu of the first cluster.
This 1 iteration becomes the column of the Phi matrix
For each input, M number of radial basis function are generated like this
'''
def GetPhiMatrix(Data, MuMatrix, BigSigma):
    DataT = Data
    TrainingLen = math.ceil(len(DataT))         
    PHI = np.zeros((int(TrainingLen),len(MuMatrix))) 
    BigSigInv = np.linalg.inv(BigSigma)
    for  C in range(0,len(MuMatrix)):
        for R in range(0,int(TrainingLen)):
            PHI[R][C] = GetRadialBasisOut(DataT[R], MuMatrix[C], BigSigInv)
    return PHI

'''
Here we are calculating the value of the basis function using the formula for
Gaussian radial basis function
'''
def GetRadialBasisOut(DataRow,MuRow, BigSigInv):    
    phi_x = math.exp(-0.5*GetScalar(DataRow,MuRow,BigSigInv))
    return phi_x

def GetScalar(DataRow,MuRow, BigSigInv):  
    R = np.subtract(DataRow,MuRow)
    T = np.dot(BigSigInv,np.transpose(R))  
    L = np.dot(R,T)
    return L

'''
Here we compute the value of the output using linear regression equation
y = w*(PHI(Transpose))
'''
def GetTestOutput(PHI,W):
    Y = np.dot(W,np.transpose(PHI))
    return Y


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

'''
sigmoid function is used in logistic regression to get the prediction in the 
range 0 and 1
'''
def GetLogisticPrediction(PHI,W):
    Y = np.dot(W,np.transpose(PHI))
    return sigmoid(Y)

''' using w = (lambda  * I + phi_T*phi)INV * t
weight matrix dimension depends on the cluster size.
we apply the Moore-Penrose pseudo-inverse of the PHI matrix to calulate the weights
'''
def GetWeightsClosedForm(PHI, T, Lambda):
    Lambda_I = np.identity(len(PHI[0]))
    for i in range(0,len(PHI[0])):
        Lambda_I[i][i] = Lambda
    PHI_T       = np.transpose(PHI)
    PHI_SQR     = np.dot(PHI_T,PHI)
    PHI_SQR_LI  = np.add(Lambda_I,PHI_SQR)
    PHI_SQR_INV = np.linalg.inv(PHI_SQR_LI)
    INTER       = np.dot(PHI_SQR_INV, PHI_T)
    W           = np.dot(INTER, T)
    return W

'''
Root mean square error is calculated as the basis of accuracy between the output
generated and the targets provided initially
'''
def GetErms(TEST_OUT,DataAct):
    sum = 0.0
    counter = 0
    for i in range (0,len(TEST_OUT)):
        sum = sum + math.pow((DataAct[i] - TEST_OUT[i]),2)
        if(int(np.around(TEST_OUT[i], 0)) == DataAct[i]):
            counter+=1
    accuracy = (float((counter*100))/float(len(TEST_OUT)))
    return np.around(math.sqrt(sum/len(TEST_OUT)),5),np.around(accuracy,5)

'''
here the cost function is calculated using the cross entropy loss function.
this measures the performance of our system. The lower value is desired.
'''
def getLogisticCost(PHI,W,t):
    #Cost = ( -log(pred) - (1-t)*log(1-pred) ) / len(t)
    pred = GetLogisticPrediction(PHI,W)
    cost = -(np.dot(np.transpose(t),np.log(pred))) - (np.dot(np.transpose((1-t)),np.log(1-pred)))
    return cost.sum() / len(t)

def getAccuracy(t,y):
    d = y - t
    acc = 1.0 - (float(np.count_nonzero(d)) / len(d))
    return acc * 100

'''
the predicted output is classified into 1 or 0 base on the threshold value 0.5
'''
def getClasses(y):
    #min = np.min(y)
    #max = np.max(y)
    y[y >= 0.5] = 1
    y[y < 0.5] = 0
    return y
    

def runLinearSGD(trainData,trainT,valData,valT,testData,testT,BigSigma,M,Lamda,learningRate,W_Lambda,epochs):
    '''
    The data is divided here into M number of clusters and the matrix containing 
    the center or mean of each cluster is calculated. The dimension if the Mu matrix
    comes out to be M x (number of features)
    '''
    #neta = [0.002,0.006,0.01,0.05,0.1]
    kmeans = KMeans(n_clusters=M, random_state=0).fit(trainData)
    Mu = kmeans.cluster_centers_
    TRAINING_PHI = GetPhiMatrix(trainData, Mu, BigSigma)
    TEST_PHI     = GetPhiMatrix(testData, Mu, BigSigma) 
    VAL_PHI      = GetPhiMatrix(valData, Mu, BigSigma)
    #W_Now        = GetWeightsClosedForm(TRAINING_PHI,trainT,W_Lambda)
    ermsArr = []
    accArr = []
    #for learningRate in neta:
    W_Now        = np.zeros((M))
    
    
    for i in range(0,epochs):
        '''
        The change in W or weight is calculated for each iteration
        by using Delta_w = neta * Delta_E and Delta_E is the derivative 
        of the error function which containes the regularizer term 
        lambda
        '''
        Delta_E_D     = -np.dot((trainT[i] - np.dot(np.transpose(W_Now),TRAINING_PHI[i])),TRAINING_PHI[i])
        La_Delta_E_W  = np.dot(Lamda,W_Now)
        Delta_E       = np.add(Delta_E_D,La_Delta_E_W)    
        Delta_W       = -np.dot(learningRate,Delta_E)
        W_T_Next      = W_Now + Delta_W
        W_Now         = W_T_Next
        
        #TEST_OUT      = GetTestOutput(TEST_PHI,W_Now)
        #[Erms_Test,Acc_Test] = GetErms(TEST_OUT,testT)
        #ermsArr.append(Erms_Test)
        #accArr.append(Acc_Test)
    
    TR_TEST_OUT   = GetTestOutput(TRAINING_PHI,W_Now) 
    VAL_TEST_OUT  = GetTestOutput(VAL_PHI,W_Now) 
    TEST_OUT      = GetTestOutput(TEST_PHI,W_Now)
    [Erms_TR,Acc_TR] = GetErms(TR_TEST_OUT,trainT)
    [Erms_Val,Acc_Val] = GetErms(VAL_TEST_OUT,valT)
    [Erms_Test,Acc_Test] = GetErms(TEST_OUT,testT)
    ermsArr.append(Erms_Test)
    accArr.append(Acc_Test)
    
    '''
    plt.plot(range(0,epochs), ermsArr) 
    plt.xlabel('Epochs') 
    plt.ylabel('ERMS for Test') 
    plt.title('Plot of ERMS error for Test set') 
    plt.show()
    
    plt.plot(range(0,epochs), accArr) 
    plt.xlabel('Epochs') 
    plt.ylabel('Accuracy for Test') 
    plt.title('Plot of Accuracy for Test set') 
    plt.show()
    '''
    #print ("Minimum Testing ERMS        = " + str(min(ermsArr)))
    #print ("Minimum Testing Accuracy    = " + str(min(accArr)))
    '''
    plt.plot(neta, ermsArr) 
    plt.xlabel('Learning rate') 
    plt.ylabel('ERMS for Test') 
    plt.title('Plot of ERMS vs learning rate for Test set') 
    plt.show()
     
    plt.plot(neta, accArr) 
    plt.xlabel('Learning rate') 
    plt.ylabel('Accuracy for Test') 
    plt.title('Plot of Accuracy vs learning rate for Test set') 
    plt.show()
     
    print("learning rates:",neta)
    print("erms array:",ermsArr)
    print("accuracy array:",accArr)
    ''' 
    #return Acc_TR,Erms_TR,Acc_Val,Erms_Val,min(accArr),min(ermsArr)
    return Acc_TR,Erms_TR,Acc_Val,Erms_Val,Acc_Test,Erms_Test

def runLogisticSGD(trainData,trainT,valData,valT,testData,testT,BigSigma,M,Lamda,learningRate,W_Lambda,epochs):
    '''
    The data is divided here into M number of clusters and the matrix containing 
    the center or mean of each cluster is calculated. The dimension if the Mu matrix
    comes out to be M x (number of features)
    '''
    #neta = [0.002,0.006,0.01,0.05,0.1]
    #kmeans = KMeans(n_clusters=M, random_state=0).fit(trainData)
    #Mu = kmeans.cluster_centers_
    #TRAINING_PHI = GetPhiMatrix(trainData, Mu, BigSigma)
    #TEST_PHI     = GetPhiMatrix(testData, Mu, BigSigma) 
    #VAL_PHI      = GetPhiMatrix(valData, Mu, BigSigma)
    TRAINING_PHI = trainData
    TEST_PHI     = testData
    VAL_PHI      = valData
    costArr = []
    accArr = []
    #W_Now        = GetWeightsClosedForm(TRAINING_PHI,trainT,W_Lambda)
    #for learningRate in neta:
    W_Now        = np.zeros((len(trainData[0])))
    
    for i in range(0,epochs):
        '''
        The change in W or weight is calculated for each iteration
        by using Delta_w = neta * Delta_E and Delta_E is the derivative 
        of the cross entropy cost function.
        '''
        pred = GetLogisticPrediction(TRAINING_PHI[i],W_Now)
        Delta_Cost    = np.dot((pred - trainT[i]),TRAINING_PHI[i])
        #Delta_Cost    = Delta_Cost / len(TRAINING_PHI)
        La_Delta_E_W  = np.dot(Lamda,W_Now)
        Delta_Cost    = np.add(Delta_Cost,La_Delta_E_W) 
        Delta_W       = -np.dot(learningRate,Delta_Cost)
        W_T_Next      = W_Now + Delta_W
        W_Now         = W_T_Next
        #TEST_OUT      = GetLogisticPrediction(TEST_PHI,W_Now)
        #TEST_OUT_CLASSES = getClasses(TEST_OUT)
        #costArr.append(getLogisticCost(TEST_PHI,W_Now,testT))
        #TEST_ACC = getAccuracy(testT,TEST_OUT_CLASSES)
        #accArr.append(TEST_ACC)
    
    TR_TEST_OUT   = GetLogisticPrediction(TRAINING_PHI,W_Now) 
    VAL_TEST_OUT  = GetLogisticPrediction(VAL_PHI,W_Now) 
    TEST_OUT      = GetLogisticPrediction(TEST_PHI,W_Now)
    TR_OUT_CLASSES = getClasses(TR_TEST_OUT)
    VAL_OUT_CLASSES = getClasses(VAL_TEST_OUT)
    TEST_OUT_CLASSES = getClasses(TEST_OUT)
    TR_ACC = getAccuracy(trainT,TR_OUT_CLASSES)
    VAL_ACC = getAccuracy(valT,VAL_OUT_CLASSES)
    TEST_ACC = getAccuracy(testT,TEST_OUT_CLASSES)
    TR_COST = getLogisticCost(TRAINING_PHI,W_Now,trainT)
    VAL_COST = getLogisticCost(VAL_PHI,W_Now,valT)
    TEST_COST = getLogisticCost(TEST_PHI,W_Now,testT)
    accArr.append(TEST_ACC)
    costArr.append(getLogisticCost(TEST_PHI,W_Now,testT))
    #print("min test cost:",min(costArr))
    '''
    plt.plot(range(0,epochs), costArr) 
    plt.xlabel('Epochs') 
    plt.ylabel('Cost for Test') 
    plt.title('Plot of Cost function for Test set') 
    plt.show()
    
    plt.plot(range(0,epochs), accArr) 
    plt.xlabel('Epochs') 
    plt.ylabel('Accuracy for Test') 
    plt.title('Plot of Accuracy for Test set') 
    plt.show()
    '''
    #print ("Minimum Testing Accuracy    = " + str(min(accArr)))
    #print ("Minimum Testing Cost    = " + str(min(costArr)))
    '''
    plt.plot(neta, costArr) 
    plt.xlabel('Learning rate') 
    plt.ylabel('Cost for Test') 
    plt.title('Plot of Cost vs learning rate for Test set') 
    plt.show()
     
    plt.plot(neta, accArr) 
    plt.xlabel('Learning rate') 
    plt.ylabel('Accuracy for Test') 
    plt.title('Plot of Accuracy vs learning rate for Test set') 
    plt.show()
     
    print("learning rates:",neta)
    print("cost array:",costArr)
    print("accuracy array:",accArr)
    '''
    return TR_ACC,TR_COST,VAL_ACC,VAL_COST,TEST_ACC,TEST_COST
    #return TR_ACC,TR_COST,VAL_ACC,VAL_COST,min(accArr),min(costArr)


def calculateSGD(data,target,TrainingPercent,ValidationPercent,TestPercent,M,Lamda,learningRate,W_Lambda,epochs,isLinear):
     
    trainingTarget = np.array(genTrainingTarget(target,TrainingPercent))
    trainingData   = genTrainingData(data,TrainingPercent)
    
    valTarget = np.array(genValAndTestTarget(target,ValidationPercent,len(trainingTarget)))
    valData   = genValAndTestData(data,ValidationPercent,len(trainingData))
    
    testTarget = np.array(genValAndTestTarget(target,TestPercent,(len(trainingTarget)+len(valTarget))))
    testData   = genValAndTestData(data,TestPercent,(len(trainingData)+len(valData)))
    
    print("Training Target Shape:",trainingTarget.shape)
    print("Training Data Shape:",trainingData.shape)
    print("Validation Target Shape:",valTarget.shape)
    print("Validation Data Shape:",valData.shape)
    print("Test Target Shape:",testTarget.shape)
    print("test Data Shape:",testData.shape)
    
    BigSigma = GenerateBigSigma(trainingData, TrainingPercent)
    
    if isLinear == True:
        [trAcc, trErms,valAcc,valErms,testAcc,testErms] = runLinearSGD(trainingData,trainingTarget,valData,valTarget,testData,testTarget,BigSigma,M,Lamda,learningRate,W_Lambda,epochs)
        print ("M = ",str(M))
        print("Lambda = ",str(Lamda))
        print("neta = ",str(learningRate))
        print ("\nTraining ERMS       = " + str(trErms))
        print ("Training Accuracy   = " + str(trAcc))
        print ("Validation ERMS     = " + str(valErms))
        print ("Validation Accuracy = " + str(valAcc))
        print ("Testing ERMS        = " + str(testErms))
        print ("Testing Accuracy    = " + str(testAcc))
    else:
        [trAcc,trCost,valAcc,valCost,testAcc,testCost] = runLogisticSGD(trainingData,trainingTarget,valData,valTarget,testData,testTarget,BigSigma,M,Lamda,learningRate,W_Lambda,epochs)
        print ("M = ",str(M))
        print("Lambda = ",str(Lamda))
        print("neta = ",str(learningRate))
        print ("Training Accuracy   = " + str(trAcc))
        print ("Training Cost   = " + str(trCost))
        print ("Validation Accuracy = " + str(valAcc))
        print ("Validation Cost = " + str(valCost))
        print ("Testing Accuracy    = " + str(testAcc))
        print ("Testing Cost    = " + str(testCost))
    
def getModel(trainingData):
    
    input_size = len(trainingData[0])
    '''
    drop_out_1 = 0.1
    drop_out_2 = 0.5
    first_dense_layer_nodes  = 64
    second_dense_layer_nodes = 32
    third_dense_layer_nodes = 1
    act1 = 'relu'
    act2 = 'relu'
    act3 = 'sigmoid'
    opt = 'adam'
    '''
    drop_out_1 = 0.5
    #drop_out_2 = 0.5
    first_dense_layer_nodes  = 128
    #second_dense_layer_nodes = 32
    third_dense_layer_nodes = 1
    act1 = 'relu'
    #act2 = 'relu'
    
    '''Sigmoid function is used for the last layer as our problem is that of 
    binary classification and we need the output as 0 or 1
    '''
    act3 = 'sigmoid'
    opt = 'rmsprop'
    
    """We need a model to store the output or the weights of the machine learning
    algorithm that would be used to determine the output of the test data.It
    basically means the classifier that we create and train to finally apply
    to the test data.
    Sequential model is simply a stack of layers and can be created
    by passing layer instances to the constructor. It is simple to use and since
    we have one source of inputs and not multiple, this is a good fit.
    """
    model = Sequential()
    
    """Dense layer declares the number of neurons, weight and biases to perform
    the linear transformation on the data. It is easy to solve and are generally
    present in most neural networks. Activation function is extremely important as 
    it decides whether a neuron is to be activated or not. It is required to perform
    non-linear transformation on the data so that it can learn and solve more
    complex tasks i.e. it is able to better fit the training data. This output is 
    passed to the next layer.
    Using activation function makes the back-propagation process possible as the
    along with errors, gradients are also sent to update the weights and biases
    appropriately.
    """
    model.add(Dense(first_dense_layer_nodes, input_dim=input_size))
    model.add(Activation(act1))
    
    """We need dropout to make sure that overfitting doesn't occur. i.e.
    some fraction of the nodes are dropped randomply so that machine learning 
    algorithm while training with a large number of epochs doesn't get biased,
    and is able to learn correctly.
    """
    model.add(Dropout(drop_out_1))       
    #model.add(Dense(second_dense_layer_nodes))
    #model.add(Activation(act2))
    #model.add(Dropout(drop_out_2))    
    model.add(Dense(third_dense_layer_nodes))
    model.add(Activation(act3))
    model.summary()
    
    """binary_crossentropy loss function is used when we have binary-class
    classification problem. In our case we have 2 classes : 0 and 1. Our aim here is to 
    minimize this loss function to improve the accuracy.
    """
    model.compile(optimizer=opt,loss='binary_crossentropy',metrics=['accuracy'])
    print("Input Dimension: ",input_size)
    print("First Dense Layer Nodes: ",first_dense_layer_nodes)
    print("First Layer activation: ",act1)
    print("First Layer dropout: ",drop_out_1)
    #print("Second Dense Layer Nodes: ",second_dense_layer_nodes)
    #print("Second Layer activation: ",act2)
    #print("Second Layer dropout: ",drop_out_2)
    print("Third Dense Layer Nodes: ",third_dense_layer_nodes)
    print("Third Layer activation: ",act3)
    print("Optimizer: ",opt)
    return model
    
def calNeuralNetwork(data,target):
    
    trainingTarget = np.array(genTrainingTarget(target,90))
    trainingData   = genTrainingData(data,90)
    
    testTarget = np.array(genValAndTestTarget(target,10,len(trainingTarget)))
    testData   = genValAndTestData(data,10,len(trainingData))
    
    model = getModel(data)
    
    '''
    10% of the data sent for training is used for validation
    '''
    validation_data_split = 0.1
    num_epochs = 1000
    print("Number of epochs: ",num_epochs)
    model_batch_size = 128
    tb_batch_size = 32
    early_patience = 100

    tensorboard_cb   = TensorBoard(log_dir='logs', batch_size= tb_batch_size, write_graph= True)
    earlystopping_cb = EarlyStopping(monitor='val_loss', verbose=1, patience=early_patience, mode='min')
    history = model.fit(trainingData
                    , trainingTarget
                    , validation_split=validation_data_split
                    , epochs=num_epochs
                    , batch_size=model_batch_size
                    , callbacks = [tensorboard_cb,earlystopping_cb]
                    ,verbose = 0
                   )
    
    df = pd.DataFrame(history.history)
    df.plot(subplots=True, grid=True, figsize=(10,15))
    plt.show()
    score = model.evaluate(testData, testTarget, batch_size=model_batch_size)
    predictedTargets = model.predict(testData) 
    predictedTargets = [round(t[0]) for t in predictedTargets] 
    print("\nNeural Network accuracy for training set: ",history.history['acc'][len(history.history['acc'])-1] * 100)
    print("Neural Network loss for training set: ",history.history['loss'][len(history.history['loss'])-1])
    print("Neural Network accuracy for validation set: ",history.history['val_acc'][len(history.history['val_acc'])-1] * 100)
    print("Neural Network loss for validation set: ",history.history['val_loss'][len(history.history['val_loss'])-1])
    print("Neural Network accuracy for Test set:",score[1]*100)
    print("Neural Network loss for Test set:",score[0])
    confmatrix = skm.confusion_matrix(testTarget,predictedTargets,labels=[1,0])
    print("\nConfusion Matrix:\n",np.array(confmatrix))
    

print ('\nUBITname      = hgarg')
print ('Person Number = 50292195')
print ('----------------------------------------------------')
print("\n\n-----PLEASE WAIT 1 HOUR FOR THE PROGRAM TO COMPLETE------\n")
    
[HODCatData,HODCatTarget,HODSubData,HODSubTarget] = getCatAndSubRawAndTargetData(HODsamePairPath,HODfeaturePath,HODdiffPairPath,'HOD')
print("HODCatData shape:",np.shape(HODCatData))
print("HODCatTarget shape:",np.shape(HODCatTarget))
print("HODSubData shape:",np.shape(HODSubData))
print("HODSubTarget shape:",np.shape(HODSubTarget))


[GSCCatData,GSCCatTarget,GSCSubData,GSCSubTarget] = getCatAndSubRawAndTargetData(GSCsamePairPath,GSCfeaturePath,GSCdiffPairPath,'GSC')
print("GSCCatData shape:",np.shape(GSCCatData))
print("GSCCatTarget shape:",np.shape(GSCCatTarget))
print("GSCSubData shape:",np.shape(GSCSubData))
print("GSCSubTarget shape:",np.shape(GSCSubTarget))


dataTime = time.time()
print("\nData load time:",dataTime-startT)

M = 25
Lamda = 2
learningRate = 0.04
W_Lambda = 0.01
epochs = 1000


"###################################### lINEAR #########################################"

print("\n\n-----Linear Regression Solution for HOD set with concatenation------")
calculateSGD(HODCatData,HODCatTarget,TrainingPercent,ValidationPercent,TestPercent,M,Lamda,learningRate,W_Lambda,epochs,True)
print("------------------------------------------------------------------------")


print("\n\n-----Linear Regression Solution for HOD set with subtraction--------")
calculateSGD(HODSubData,HODSubTarget,TrainingPercent,ValidationPercent,TestPercent,M,Lamda,learningRate,W_Lambda,epochs,True)
print("------------------------------------------------------------------------")


print("\n\n-----Linear Regression Solution for GSC set with concatenation------")
calculateSGD(GSCCatData,GSCCatTarget,TrainingPercent,ValidationPercent,TestPercent,M,Lamda,learningRate,W_Lambda,epochs,True)
print("------------------------------------------------------------------------")

print("\n\n-----Linear Regression Solution for GSC set with subtraction--------")
calculateSGD(GSCSubData,GSCSubTarget,TrainingPercent,ValidationPercent,TestPercent,M,Lamda,learningRate,W_Lambda,epochs,True)
print("------------------------------------------------------------------------")

"#######################################################################################"



"###################################### LOGISTIC #######################################"

print("\n\n-----Logistic Regression Solution for HOD set with concatenation----")
calculateSGD(HODCatData,HODCatTarget,TrainingPercent,ValidationPercent,TestPercent,M,Lamda,learningRate,W_Lambda,epochs,False)
print("------------------------------------------------------------------------")


print("\n\n-----Logistic Regression Solution for HOD set with subtraction------")
calculateSGD(HODSubData,HODSubTarget,TrainingPercent,ValidationPercent,TestPercent,M,Lamda,learningRate,W_Lambda,epochs,False)
print("------------------------------------------------------------------------")


print("\n\n-----Logistic Regression Solution for GSC set with concatenation----")
calculateSGD(GSCCatData,GSCCatTarget,TrainingPercent,ValidationPercent,TestPercent,M,Lamda,learningRate,W_Lambda,epochs,False)
print("------------------------------------------------------------------------")

print("\n\n-----Logistic Regression Solution for GSC set with subtraction------")
calculateSGD(GSCSubData,GSCSubTarget,TrainingPercent,ValidationPercent,TestPercent,M,Lamda,learningRate,W_Lambda,epochs,False)
print("------------------------------------------------------------------------")

"#######################################################################################"



"####################################### NEURAL ########################################"

print("\n\n-----Neural Network Solution for HOD set with concatenation----")
calNeuralNetwork(HODCatData,HODCatTarget)
print("------------------------------------------------------------------------")


print("\n\n-----Neural Network Solution for HOD set with subtraction------")
calNeuralNetwork(HODSubData,HODSubTarget)
print("------------------------------------------------------------------------")


print("\n\n-----Neural Network Solution for GSC set with concatenation----")
calNeuralNetwork(GSCCatData,GSCCatTarget)
print("------------------------------------------------------------------------")

print("\n\n-----Neural Network Solution for GSC set with subtraction------")
calNeuralNetwork(GSCSubData,GSCSubTarget)

print("------------------------------------------------------------------------")

"#######################################################################################"


endT = time.time()
print("Elapsed time:",endT-startT)