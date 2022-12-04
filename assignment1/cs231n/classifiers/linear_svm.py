from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

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
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    # C: classes
    num_classes = W.shape[1]
    # N: train number
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        # scores: shape(C)
        scores = X[i].dot(W)
        # correct score
        # y[i] = c means that X[i] has label c, where 0 <= c < C
        correct_class_score = scores[y[i]]
        # L_i = sum(max(0, s_j-s_y_i+1)) (j != y_i)
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if margin > 0:
                loss += margin
                dW[:,y[i]] += -X[i,:]
                dW[:,j] += X[i,:]

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    # L = sum(L_i)/N
    loss /= num_train

    # Add regularization to the loss.
    # 加入正则项 lambda*R(W)
    loss += reg * np.sum(W * W)

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # pass
    
    dW /= num_train
    dW += reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # pass
    
    # scores: shape(N, C)
    scores = X.dot(W)
    # N: train number
    num_train = X.shape[0]
    # 第一个参数表示取行的范围，np.arange(num_train)=500，即取所有行（总共行为500）
    # 第二个参数表示取列
    # 所以就是取0行的多少列，1行的多少列，2行的多少列，最终得到每张图片，正确标签对应的分数
    correct_scores = scores[np.arange(num_train), y]
    correct_scores = correct_scores.reshape((num_train, -1))
    # 计算每个误差
    margins = np.maximum(0, scores-correct_scores+1)
    # 将label值所在的位置置零
    margins[range(num_train), y] = 0
    loss += np.sum(margins)
    loss /= num_train
    loss += reg * np.sum(W*W)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # pass

    # 将margins>0的项（有误差的项）置为1，没误差的项为0
    margins[margins > 0] = 1

    # 没误差的项中有一项为标记项，计算标记项的权重分量对误差也有共享，也需要更新对应的权重分量
    # margins中这个参数就是当前样本结果错误分类的数量
    row_num = -np.sum(margins,1)
    margins[np.arange(num_train), y] = row_num
    
    # X: 200x3073    margins:200x10  -> 10x3072
    dW += np.dot(X.T, margins)  # 3073x10
    dW /= num_train  # 平均权重
    dW += reg * W  # 正则化

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW