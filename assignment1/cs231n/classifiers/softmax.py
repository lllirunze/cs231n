from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass
    
    # N: 获取样本数量
    num_train = X.shape[0]
    # C: 获取样本类别
    num_classes = W.shape[1]
    
    for i in range(num_train):
        # score_i.shape = [1, C]
        score_i = X[i].dot(W)
        exp_score_i = np.exp(score_i)
        sum_score_i = np.sum(exp_score_i)
        exp_score_i /= sum_score_i
        
        correct_score_i = exp_score_i[y[i]]
        # 计算样本的损失
        loss += (-np.log(correct_score_i))
        # 计算梯度
        for j in range(num_classes):
            if j != y[i]:
                dW[:,j] += exp_score_i[j]*X[i]
            else:
                dW[:,j] += (exp_score_i[y[i]]-1)*X[i]
                
    loss /= num_train
    loss += reg*np.sum(W*W)
    dW /= num_train
    dW += reg*W
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass

    # N: 获取样本数量
    num_train = X.shape[0]
    
    # scores.shape: [N, C]
    scores = X.dot(W)
    exp_scores = np.exp(scores)
    # sum_scores.shape: [N,]
    sum_scores = np.sum(exp_scores, axis=1)
    exp_scores /= sum_scores[:, np.newaxis]
    
    # 计算样本损失
    loss_matrix = -np.log(exp_scores[range(num_train),y])
    loss += np.sum(loss_matrix)
    # 计算梯度
    # 取正确标签处，减一。注：必须使用range(num_train)而非':'
    exp_scores[range(num_train),y] -= 1
    dW += np.dot(X.T, exp_scores)  # 3073x500 * 500x10 = 3073x10

    loss /= num_train
    loss += reg * np.sum(W*W)
    dW /= num_train
    dW +=reg *W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
