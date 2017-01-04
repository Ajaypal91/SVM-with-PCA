# File Name: pca_svd_dimension_reduction.py
# By: Leonard Wesley
# Date: May 21, 2016
# Python Version(s) 2.7.10 | 64-bit: | Enthought Canopy (default, Oct 21 2015, 17:08:47) [MSC v.1500 64 bit (AMD64)]'

__VERSION = 1.0

### import section
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn import preprocessing
import os, csv
import numpy as np
### end of import section 

### CONSTANTS:
__INFINITY = float('inf')
### end of CONSTANTS

### Global variable definition section 
__DEFAULT_PCA_SVD_COVARIANCE_THRESHOLD = 0.7
__PCA_SVD_COVARIANCE_THRESHOLD = __DEFAULT_PCA_SVD_COVARIANCE_THRESHOLD
__PCA_SVD_COVARIANCE_THRESHOLD = __PCA_SVD_COVARIANCE_THRESHOLD
### End Global variable definition section 

### Helper function definition section
def getSVD(X,numberOfDimensions=60) :
    print "====================== Performing SVD ========================="
    print ">>>>> num_dimensions=", numberOfDimensions
            
    print "Starting to fit SVD"
    svd = TruncatedSVD(n_components = numberOfDimensions) 
    svd.fit(X)
    return svd

def getSvdComponents(svd) :
    svdComp = svd.components_
    ind = range(svdComp.shape[1])
    s = np.concatenate((np.array(ind).reshape(len(ind),1), np.array([sum(svdComp[:,i]) for i in ind]).reshape(len(ind),1)),axis=1)
    #array with index and features PC loading factors values of features in decreasing order
    s = s[np.argsort(s[:,1])][::-1]
    #print s
    return ind,s
    
def reduce_dimension(svd,X): 
    '''
    X_shape = X.shape 
    
    if X_shape[0] == X_shape[1]:  # If number samples equals dimension use PCA else use SVD 
        print
        print "The dimension of the data set is square, and PCA can be used."
        print "However, PCA is not implemented yet, so SVD will be used."
        print
    '''     
    #print "\n <------------ Reducing Dimensions ------------> \n"
    X_reduced = svd.transform(X)
    #X_reduced  = svd_trans.transform(X)
    print "Reduced Shape " , X_reduced.shape
    return X_reduced

