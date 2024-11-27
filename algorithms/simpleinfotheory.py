'''
migrated definitions from module 1
'''
import numpy as np
from typing import Tuple, Iterable


_DELTA = 0.001  # acceptable rounding error in probability sum


def _probability_1d(xn) -> Tuple[Iterable[object], Iterable[float]]:
    '''
    input:
        xn: iterable of elements
    output:
        symbols: unique symbols
        probabilities: probabilities corresponding to symbols
    '''
    if not isinstance(xn, np.ndarray):
        xn = np.array(xn)
    symbols, xn_count = np.unique(xn, return_counts=True)
    probabilities = xn_count / len(xn)
    return symbols, probabilities


def _probability_2d(xn, yn) -> Tuple[Iterable[object], Iterable[float]]:
    '''
    compute p(xi, yj), of xi in xn, yj in yn
    input:
        xn: iterable of elements
        yn: iterable of elements
    output:
        symbols: list of (xi, yj) combinations
        probabilities: corresponding probabilities p(xi, yj)
    '''
    # create a pandas dataframe, with unique 
    if not isinstance(xn, np.ndarray):
        xn = np.array(xn)
    if not isinstance(yn, np.ndarray):
        yn = np.array(yn)
    xnyn = np.row_stack((xn, yn)).T
    symbols, xnyn_count = np.unique(xnyn, axis=0, return_counts=True)
    probabilities = xnyn_count / np.sum(xnyn_count)
    assert sum(probabilities) - 1 < _DELTA
    return symbols, probabilities


def infocontent(p):
    # Alter the equation below to provide the correct Shannon information 
    # content:
    return - np.log2(p)


'''
module 2, about entropy, entropy empirical, joint entropy
'''


def entropy(p) -> float:
    '''
    entropy of single-dimension probabilities p
    input:
        p: list[float], list of probabilities sum to 1
    output:
        H_sum, H(X) of probabilities p(x=X)
    '''
    # First make sure the array is now a numpy array
    if isinstance(p, float) or isinstance(p, int):
        if p > 1 or p < 0:
            raise ValueError('probability shoud be between 0 and 1')
        p = [p, 1 - p]
    if isinstance(p, np.ndarray):
        probabilities = p
    else:
        probabilities = np.array(p)
    # Should we check any potential error conditions on the input?
    assert np.abs(np.sum(probabilities) - 1) < 0.0001  # sum rounding to 1

    # We need to take the expectation value over the Shannon info content at
    # p(x) for each outcome x:
    # Alter the equation below to provide the correct entropy:
    Hs = []
    for ps in probabilities:
        if ps == 0:
            Hs.append(0)
        else:
            Hs.append(- ps * np.log2(ps))
    H_sum = np.sum(Hs)
    return H_sum


def entropyempirical(xn):
    '''
    Computes the joint Shannon entropy over all outcome vectors x of a vector
    random variable X with probability matrix p(x) for each candidate outcome
    vector x.

    inputs:
    xn
    outputs:
    result: emtropy H(X)
    symbols: unique symbols in xn
    probabilities: corresponding probabilities
    '''
    symbols, probabilities = _probability_1d(xn)
    result = entropy(probabilities)
    return result, symbols, probabilities


def jointentropy(p):
    '''
    H(X,Y), joint entropy of X and Y
    inputs:
    p: [p(x1, y1), ...], list of lists, a matrix of probabilities
    returns:
    joint_entropy, float
    '''

    p_flat = np.asarray(p).flatten()
    H_XY = entropy(p_flat)
    # joint_entropy = sum([p * entropy(p_flat) for p in p_flat])
    # return joint_entropy
    return H_XY


def jointentropyempirical(xn, yn):
    '''
    inputs:
    xnyn: list[list], values of x and y, [[x1,y1],[x2,y2],...]
    returns:
    jointentropyempirical, float, joint entropy of xnyn samples
    '''
    symbols, probabilities = _probability_2d(xn, yn)
    result = entropy(probabilities)
    return result, symbols, probabilities


'''
module 2, conditional entropy
'''


def conditionalentropy(p):
    """
    Inputs:
    - p - 2D probability distribution function over all outcomes (x,y).
    p is a matrix over all combinations of x and y,
    where p(1,3) gives the probability of the first symbol of variable
    x co-occuring with the third symbol of variable y.
    E.g. p = [0.2, 0.3; 0.1, 0.4]. The sum over p must be 1.

    Outputs:
    - result - conditional Shannon entropy of X given Y
    """
    # First make sure the array is now a numpy array
    if not isinstance(p, np.ndarray):
        p = np.array(p)

    # Check that the probabilities normalise to 1:
    if (abs(np.sum(p) - 1) > 0.00001):
        raise Exception("Probability distribution must sum to 1: sum is %.4f"
                        % np.sum(p))

    # We need to compute H(X,Y) - H(X):
    # 1. joint entropy of X and Y
    H_XY = jointentropy(p)
    # 2. marginal entropy of Y
    #  But how to get p_y???
    p_y = p.sum(axis=0)  # Since y changes along the columns, summing over the x's (dimension 0 argument in the sum) will just return p(y)
    H_Y = entropy(p_y)
    result = H_XY - H_Y
    return result


def conditionalentropyempirical(xn, yn):
    '''
    inputs:
        xn, yn
    outputs:
        H_X, given Y
    '''
    # First, error checking, and converting argument into standard form:    
    xn = np.array(xn)
    yn = np.array(yn)

    # # Convert to column vectors if not already:
    # if xn.ndim == 1:
    #     xn = np.reshape(xn, (len(xn), 1))
    # yn = np.array(yn)
    # if yn.ndim == 1:
    #     yn = np.reshape(yn, (len(yn), 1))
    # # check that their number of rows are the same
    assert xn.shape[1] == yn.shape[1]

    # We need to compute H(X,Y) - H(X):
    # 1. joint entropy: Can we re-use existing code?
    (H_XY, xySymbols, xyProbs) = jointentropyempirical(xn, yn)
    # 2. marginal entropy of Y: Can we re-use existing code?
    (H_Y, ySymbols, yProbs) = entropyempirical(yn)
    result = H_XY - H_Y
    return result


'''
module 3, mutual information
'''


def mutualinformation(p) -> float:
    '''
    inputs:
        p: probability matrix of x and y
    outputs:
        mutual: mutual information I(X;Y)
    '''
    # First make sure the array is now a numpy array
    if type(p) != np.array:
        p = np.array(p)
    if (abs(np.sum(p) - 1) > 0.0001):
        raise Exception(
            f"Probability distribution must sum to 1, but is {np.sum(p)}")
    # We need to compute H(X) + H(Y) - H(X,Y):
    # 1. joint entropy:
    H_XY = jointentropy(p)

    # 2. marginal entropy of X:
    # But how to get p_x???
    p_x = p.sum(axis=1)
    H_X = entropy(p_x)

    # 2. marginal entropy of Y:
    # But how to get p_y???
    p_y = p.sum(axis=0)
    H_Y = entropy(p_y) 

    mutual = H_X + H_Y - H_XY
    return mutual


def mutualinformationempirical(xn, yn) -> float:
    '''
    inputs:
        xn, yn: values of x and y
    outputs:
        mutual: mutual information I(X;Y)
    '''
    xn = np.array(xn)
    yn = np.array(yn)
    assert xn.shape[0] == yn.shape[0]
    (H_XY, xySymbols, xyProbs) = jointentropyempirical(xn, yn)  # array [[x1,y1],[x2,y2],...]
    # 2. marginal entropy of Y: (call 'joint' in case yn is multivariate)
    (H_Y, ySymbols, yProbs) = entropyempirical(yn)
    # 3. marginal entropy of X: (call 'joint' in case yn is multivariate)
    (H_X, xSymbols, xProbs) = entropyempirical(xn)
    # 4. apply the equation I(X;Y) = H(x) + H(Y) - H(X,Y)
    result = H_X + H_Y - H_XY
    return result
