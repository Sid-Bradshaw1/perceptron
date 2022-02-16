  
import numpy as np
import time
from random import Random
from sklearn.base import BaseEstimator, ClassifierMixin

### NOTE: The only methods you are required to have are:
#   * predict
#   * fit
#   * score
#   * get_weights
#   They must take at least the parameters below, exactly as specified. The output of
#   get_weights must be in the same format as the example provided.

from sklearn.linear_model import Perceptron

class PerceptronClassifier(BaseEstimator,ClassifierMixin):

    def __init__(self, lr=.1, shuffle=False, Deterministic=10, sc=.01):
      """ Initialize class with chosen hyperparameters.
      Args:
          lr (float): A learning rate / step size.
          shuffle: Whether to shuffle the training data each epoch. DO NOT SHUFFLE for evaluation / debug datasets.
      """
      self.epochs = Deterministic
      self.lr = lr
      self.shuffle = shuffle
      self.sc = sc
      self.misses = [] #lol didnt need this
      self.stopped_epoch = 0

    def fit(self, X, y, initial_weights=None):
        """ Fit the data; run the algorithm and adjust the weights to find a good solution
        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets
            y (array-like): A 2D numpy array with the training targets
            initial_weights (array-like): allows the user to provide initial weights
        Returns:
            self: this allows this to be chained, e.g. model.fit(X,y).predict(X_test)
        """

       #x and y are numpy arrays already
       #add a check to make sure the initial weights match the appropriate dimension 
        n = len(y)
        if self.shuffle:
          X, y = self._shuffle_data(X, y)
      
        
        #augment pattern with 1s
        pattern = self.augment_ones(X)
        

        targets = y

        self.weights = self.initialize_weights(pattern) if not initial_weights else initial_weights
      

        
        
        
        #change 1 to self.epochs when done testing
        #make vec of 0s these are the change in weights

        delta_dub = np.zeros(len(self.weights))
        
        for j in range(10):
            #count miss classifications per epoch
            miss = 0
            
            # print(j)
            
            for i in range(len(pattern)):
              self.weights = np.add(self.weights, delta_dub)
              
              myVec = []
              for k in range(len(pattern[i])):
                myVec.append(pattern[i][k])
              myVec = np.array(myVec)
              
              net = np.dot(myVec, self.weights)
              
              if net > 0:
                output = 1
              else:
                output = 0
              if targets[i] != net:
                miss += 1
              self.misses.append(miss/n)
              delta_dub = self.lr * (targets[i] - output)*pattern[i]
              
            # print(self.weights)
            
            

           

      
        return self

    def predict(self, X):
        """ Predict all classes for a dataset X
        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets
        Returns:
            array, shape (n_samples,)
                Predicted target values per element in X.
        """
        np_X = np.array([np.append(x, 1) for x in X])
        nets = np.dot(np_X, self.weights.T)
        return np.where(nets > 0,1,0)

    def augment_ones(self, arr):
        
        
        a = np.ones((len(arr), 1))
        arr = np.hstack((arr, a))
        return arr

    def initialize_weights(self,shapes):
        """ Initialize weights for perceptron. Don't forget the bias!
        Returns:
        """
        apple = np.zeros(len(shapes[0]))
        return apple

    def score(self, X, y, s=False):
        """
        Return accuracy of model on a given dataset. Must implement own score function.
        Args:
            X (array-like): A 2D numpy array with data, excluding targets
            y (array-like): A 2D numpy array with targets
        Returns:
            score : float
                Mean accuracy of self.predict(X) wrt. y.
        """

        predictions = self.predict(X)
        
        return sum(predictions == y) / len(y)

    def _shuffle_data(self, X, y):
        """
        Shuffle the data!
        """
        shuffled = np.random.permutation(range(len(y)))
        return X[shuffled], y[shuffled]

    def split_func(self, X, y, percentage):
      temp = np.random.rand(X.shape[0])
      split_matrix = temp < np.percentile(temp, percentage)

      training_x = X[split_matrix]
      training_y = y[split_matrix]
      testing_x = X[~split_matrix]
      testing_y = y[~split_matrix]
      return training_x, training_y, testing_x, testing_y

    def get_weights(self):
        return self.weights