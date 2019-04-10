import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_classes = W.shape[1]
  num_dim = W.shape[0]
  num_train = X.shape[0]
  loss = 0.0
  for i in range(num_train):
    scores = X[i].dot(W)
    exp = np.exp(scores)
    normalize = exp/np.sum(exp)
    loss += -np.log(normalize[y[i]])
    normalize[y[i]] -= 1 # subracting y[i]=1 when classes match
    #dW += np.dot(np.reshape(X[i],(num_dim,1)), np.reshape(normalize,(1,num_classes)))     
    #dW is adjusted by each row being the X[i] pixel values by the probability (normalize) vector
    for j in range(num_classes):
        dW[:,j] += X[i] * normalize[j]    

    
  # Normalize
  loss /= num_train
  dW /= num_train
  # Regularize
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_classes = W.shape[1]
  num_train = X.shape[0]

  scores = X.dot(W)
  lim_scores = scores - np.max(scores) # Limit scores
  # Array of correct scores
  correct_scores = lim_scores[np.arange(num_train), y]
  # Compute loss
  loss_array = - correct_scores + np.log(np.sum(np.exp(lim_scores), axis=1))
  loss = np.sum(loss_array)

  softmaxes = np.exp(lim_scores) / np.sum(np.exp(lim_scores), axis=1)[:,None]
  # Matrix A size (num_train, num_classes), same purpose as in SVM
  # A[sample, correct_class] = softmax-1
  # A[sample, incorrect_class] = softmax  
  A = np.zeros([num_train, num_classes]) 
  A[np.arange(num_train), y] = -1
  A += softmaxes
  # Compute gradient
  dW = X.T.dot(A)

  # Normalize
  loss /= num_train
  dW /= num_train
  # Regularize
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW
