"""
information dynamics methods

outline:
- multivariate joint entropy, gaussian estimator

"""
import typing
import numpy as np


# MULTIVARIATE JOINT ENTROY

def discrete_entropy(data: typing.Iterable) -> float:
    """
    calulate discrete entropy of data

    inputs:
        - `data`: iterable, containing discrete values or symbols
    outputs:
        - `entropy`: float, entropy value in nats.
    """
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    _, counts = np.unique(data, axis=0, return_counts=True)
    probabilities = counts / np.sum(counts)
    entropy = - np.sum(np.dot(probabilities, np.log(probabilities)))
    return entropy


def cov_det(data: np.ndarray) -> float:
    """
    Computes Big-Sigma covariance determinant inside
    Joint Entropy Gaussian Estimator.

    inputs:
    - data: np.ndarray, rows correspond to dimensions.

    outputs:
    - det: float, determinant of Cov[xi,yj]
    """
    cov_matrix = np.cov(data, rowvar=True)
    det = np.linalg.det(cov_matrix)
    return det


def gaussian_joint_entropy(data: np.ndarray) -> float:
    """
    Computes Joint Entropy of multivariate data
    using Gaussian Estimator.

    inputs:
    - data: np.ndarray, rows correspond to dimensions.

    outputs:
    - entropy: float, joint entropy in nats.
    """
    big_sigma = np.cov(data, rowvar=True)
    # calculate pseudo determinant
    # pseudo determinant: product of non-zero eigenvalues
    eigenvalues = np.linalg.eigvalsh(big_sigma)
    threshold = 1e-10
    nonzero_engenvalues = eigenvalues[eigenvalues > threshold]
    # handle all-zero determinant (big sigma)
    if len(nonzero_engenvalues) == 0:
        return 0.0
    # determinant equals products
    determinant = np.prod(nonzero_engenvalues)
    # apply gaussian estimator equation
    dimensions = data.shape[0] if np.ndim(data) > 1 else len(data)
    joint_entropy = dimensions / 2 * (
        1 + np.log(2 * np.pi) + np.log(determinant))
    return joint_entropy


def discrete_joint_entropy(
        data: np.ndarray,
        ) -> float:
    """
    calculate discrete, multivariate joint entropy

    inputs:
        - `data`: numpy array, or list of arrays,
        rows correspond to dimensions.
    outputs:
        - `joint_entropy`: float, entropy in nats
    """
    _, counts = np.unique(data, axis=0, return_counts=True)
    probabilities = counts / np.sum(counts)
    joint_entropy = discrete_entropy(probabilities)
    return joint_entropy


# dictionary of joint entropy methods
joint_entropy_methods = {
    'gaussian': gaussian_joint_entropy,
    'discrete': discrete_joint_entropy,
}


# INFORMATION DYNAMICS

def shift_array(array: np.ndarray, shifts: int) -> np.ndarray:
    """
    inputs:
    - array: 1-dimensional numpy array
    - shifts: int, number of shifts.
    positive shifts to right, negative to left.

    outputs:
    - result: shifted array
    """
    if not np.ndim(array) == 1:
        raise ValueError()
    # shift np array like pd.Series.shift
    result = np.empty_like(array, dtype=float)
    if shifts > 0:
        result[:shifts] = np.nan
        result[shifts:] = array[:-shifts]
    elif shifts < 0:
        result[shifts:] = np.nan
        result[:shifts] = array[-shifts:]
    elif shifts == 0:
        result = array
    return result


def entropy_rate(x: np.ndarray, k: int = None, estimator='discrete') -> float:
    """
    Computes the mutual information of 

    inputs:
    data: 1-dimensional array

    outputs:
    entropy_rate: float
    """
    if not isinstance(k, int):
        k = len(x) - 3
    x_k = [shift_array(x, s) for s in range(k+1)]
    x_k_0 = np.row_stack(x_k)  # sum to n
    x_k_1 = x_k_0[1:]  # sum to n-1
    # filter columns containing nan
    x_k_0 = x_k_0[:, ~np.isnan(x_k_0).any(axis=0)]
    x_k_1 = x_k_1[:, ~np.isnan(x_k_1).any(axis=0)]
    joint_entropy_estimator = joint_entropy_methods[estimator]
    joint_0 = joint_entropy_estimator(x_k_0)
    joint_1 = joint_entropy_estimator(x_k_1)
    er = joint_1 - joint_0
    return er


if __name__ == '__main__':
    # test_array = np.arange(1, 10)
    base = np.sin(np.arange(0, 2 * np.pi, 0.1 * np.pi))
    fluctuation = np.random.normal(0, 0.1, size=20)
    displacement = np.cumsum(fluctuation)
    test_array = base + displacement
    # shift_array(test_array, 3)
    base_er = entropy_rate(base)
    print('base:', base_er)
    displacement_er = entropy_rate(displacement)
    print('displacement:', displacement_er)
    er = entropy_rate(test_array)
    print('base + displacement:', er)
    print('debug...')
