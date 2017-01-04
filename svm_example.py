import csv, sys, os
import numpy as np
from sklearn import svm, preprocessing
import myPCA as mp
import matplotlib.pyplot as plt
reload(mp)


#CONSTANTS:
__INFINITY = float('inf')

def make_sure_isnumber(n, row_index, col_index, compound, header, nan_locs):

    try:
        # If number is > infinity then return np.nan that will need to be cleaned after dataset is completely read in.
        if n == np.nan  or  float(n) >= __INFINITY  or  float(n) == np.nan:
            print  "*** Encountered value =", n, " for the compound named ", compound," and descriptor named ", header[col_index]
            nan_locs = nan_locs.append((row_index, col_index))
            return np.nan
        return float(n)  # else return the number as a float. 
    except ValueError:
        return 0.

def loadTrainingData() :
    try:
        cr = csv.reader(open("BRAF_train_moe_class.csv"))
    except IOError:
        raise IOError("Problems locating or opening the .csv file named 'BRAF_train_moe_class.csv' were encountered.")

    # Save the first row that is a header row and convert it to a numpy array
    header = np.array(cr.next()) 
    
    # Read in the rest of data, and convert items after 3rd col from string to number.
    # Assume col 0 is the compound name, col 1 is CID=Compound ID, and 
    # col 2 contains the class info
    data = np.arange(0)  # Create an empty 1D array that will hold the training data.
    
    # Extract column header names starting from 4th column and higher
    data_header = header[3:]
    
    # List of (row, col) coordinates of np.nan values to be cleaned later
    # nan_locs is a mutable list and is modified by the make_sure_isnumber  function if and when its first argument
    # is >= infinity or a nan. If  thsi is the case, nan_locs is appended with the list [row, col] that is the row and column in
    # the dataset that will need to be cleaned later. 
    nan_locs = []   
    row_index = 0
    
    for row in cr:
        data_row = row[3:]
        new_data_row = np.arange(0)
    
        if len(data_header) == len(data_row): 
            for col_index in range(len(data_header)):
                    new_data_row = np.concatenate((new_data_row, [(make_sure_isnumber(data_row[col_index], \
                                    row_index, col_index, row[0], data_header, nan_locs))])) 
                
            if len(data) > 0:
                data = np.vstack((data, np.concatenate((row[:3], new_data_row))))
            else:
                data = np.concatenate((row[:3], new_data_row))
    
    
    class_info = np.array(map(lambda x: int(float(x)), data[:,2]))
    #class_info = np.array([int(x) for x in data[:,2] ])
    
    # Make sure class is either 0 or 1. If so, append it to class_info list else raise error.
    for c in class_info:
        if c not in [0,1]:
            raise ValueError("The column named ",header[2], " in example_svm_train.csv has a value not equal to 0 or 1.")
    
    X = np.array(data[:,3:], dtype = float)
    X = preprocessing.scale(X)   #  scale data between [-1, 1]
    y = np.array(class_info, dtype = int)
    print "shape of training data = " , X.shape
    return X,y

def loadTestingData() :
    try:
        cr = csv.reader(open("BRAF_test_moe_class.csv"))
    except IOError:
        raise IOError("Problems locating or opening the .csv file named 'BRAF_train_moe_class.csv' were encountered.")
    
    
    header = np.array(cr.next()) 
    
    data = np.arange(0)  # Create an empty 1D array that will hold the training data.
    
    data_header = header[3:]
    
    nan_locs = []   
    row_index = 0
    
    for row in cr:
        data_row = row[3:]
        new_data_row = np.arange(0)
    
        if len(data_header) == len(data_row): 
            for col_index in range(len(data_header)):
                    new_data_row = np.concatenate((new_data_row, [(make_sure_isnumber(data_row[col_index], \
                                    row_index, col_index, row[0], data_header, nan_locs))])) 
                
            if len(data) > 0:
                data = np.vstack((data, np.concatenate((row[:3], new_data_row))))
            else:
                data = np.concatenate((row[:3], new_data_row))
    
    class_info = np.array(map(lambda x: int(float(x)), data[:,2]))
    #class_info = np.array([int(x) for x in data[:,2] ])
    
    # Make sure class is either 0 or 1. If so, append it to class_info list else raise error.
    for c in class_info:
        if c not in [0,1]:
            raise ValueError("The column named ",header[2], " in example_svm_train.csv has a value not equal to 0 or 1.")
    
    
    X = np.array(data[:,3:], dtype = float)
    X = preprocessing.scale(X)   #  scale data between [-1, 1]
    y = np.array(class_info, dtype = int)
    print "shape of testing data = " , X.shape
    return X,y

def getClassifier(X,y) :
    return svm.SVC(kernel='rbf', C=1, gamma = 'auto', degree = 3.0, coef0 = 0.0).fit(X, y)

def getAccuracy(clf,X,y) :
    score = format(int((clf.score(X, y) * 10000))/100.)
    print "clf.score(X, y) = %s"  %score
    #print "clf.predict(X) (clf is classifier of SVM) = ", clf.predict(X)
    return score
 
def plotGraph(ind,s) :
    
    plt.plot(ind,s[:,1])
    plt.ylabel("Loading factor") 
    
def plotElbow(xVal) :
    x = [xVal] * 5
    y = range(-2,3)
    plt.plot(x,y)   
       
#def returnNoOfFeatures(svdComp, var) :
#    v = 0
#    i = 0
#    while v <= var :
#        v += svdComp[i,1]
#        i += 1
#    return i+1            
    
def __init__(tolerance) :
    
    Xtrain,ytrain = loadTrainingData()
    Xtest,ytest = loadTestingData()
    clf = getClassifier(Xtrain,ytrain)                
    reduceddimensions = 60
    #reading testing file
    
    maxScore = getAccuracy(clf,Xtest,ytest)
    score = maxScore
    print "Accuracy with all features " , maxScore
    
    #get Svd
    svd = mp.getSVD(Xtrain,reduceddimensions)
    ind,svdComp  = mp.getSvdComponents(svd)
        
    #max_var = max(svdComp[:,1])
    #min_var = min(svdComp[:,1])
    
    #print max_var, min_var
    variance_to_acc_for = sum(svdComp[0:reduceddimensions,1])
    print "Maximum variance accounted ", variance_to_acc_for
        
    i = 1    
    #noOfF = returnNoOfFeatures(svdComp,variance_to_acc_for)
    #print "Number of features %s" ,noOfF
    
    plt.subplot(5,1,i)
    plt.title("Elbow Plots")
    plotGraph(ind, svdComp)
    plotElbow(reduceddimensions)  
    text = "Accuracy = %s \nReduced Features = %s" %(score,reduceddimensions)
    plt.text(250,0.5,text, fontdict={"fontsize":"10"})  

    while i < 6 :
        print "reducing dimensions to %s" , reduceddimensions 
        
        #number of plots, x, y in grid
           
        Xtrain_reduced = mp.reduce_dimension(svd, Xtrain)
        Xtest_reduced = mp.reduce_dimension(svd, Xtest)
        clf = getClassifier(Xtrain_reduced,ytrain)    
        score = getAccuracy(clf,Xtest_reduced,ytest)
    
        if float(score) < (float(maxScore) - tolerance) :
            break
        reduceddimensions -= 5   
        svd = mp.getSVD(Xtrain,reduceddimensions)
        #svdComp  = mp.getSvdComponents(svd) 
        i += 1
        plt.subplot(5,1,i)
        plotGraph(ind, svdComp)
        plotElbow(reduceddimensions)
        text = "Accuracy = %s \nReduced Features = %s" %(score,reduceddimensions)
        plt.text(250,0.5,text, fontdict={"fontsize":"10"})  
           
    plt.xlabel("Feature number")
    plt.show()
    print "\n\nBest number of Fit Features " , reduceddimensions+5    
        
    
            
__init__(5)             







