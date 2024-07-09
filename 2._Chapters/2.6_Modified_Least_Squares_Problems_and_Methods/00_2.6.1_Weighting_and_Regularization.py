# 00_2.6.1_Weighting_and_Regularization

"""

Lecture: 2._Chapters/2.6_Modified_Least_Squares_Problems_and_Methods
Content: 00_2.6.1_Weighting_and_Regularization

"""

import numpy as np
from typing import Tuple

def weighted_least_squares_row_weight(A: np.ndarray, b: np.ndarray, D: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    使用行权重解决加权最小二乘问题

    Args:
    - A (np.ndarray): 系数矩阵 A，形状为 (m, n)
    - b (np.ndarray): 右侧向量 b，形状为 (m,)
    - D (np.ndarray): 对角行权重矩阵 D，形状为 (m, m)

    Returns:
    - x (np.ndarray): 解向量 x，形状为 (n,)
    - residual_norm (float): 残差范数 \|Ax - b\|_2
    """
    weighted_A = D @ A
    weighted_b = D @ b
    x = np.linalg.lstsq(weighted_A, weighted_b, rcond=None)[0]
    residual = A @ x - b
    residual_norm = np.linalg.norm(residual, ord=2)
    return x, residual_norm

# 示例用法
def test1():
    A = np.array([[1, 2], [3, 4], [5, 6]])
    b = np.array([1, 2, 3])
    D = np.diag([0.5, 1.0, 1.5])
    x, residual_norm = weighted_least_squares_row_weight(A, b, D)
    print("解向量 x:", x)
    print("残差范数:", residual_norm)

def weighted_least_squares_column_weight(A: np.ndarray, b: np.ndarray, G: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    使用列权重解决加权最小二乘问题

    Args:
    - A (np.ndarray): 系数矩阵 A，形状为 (m, n)
    - b (np.ndarray): 右侧向量 b，形状为 (m,)
    - G (np.ndarray): 非奇异列权重矩阵 G，形状为 (n, n)

    Returns:
    - x (np.ndarray): 解向量 x，形状为 (n,)
    - residual_norm (float): 残差范数 \|Ax - b\|_2
    """
    weighted_A = A @ np.linalg.inv(G)
    weighted_b = b
    x = np.linalg.lstsq(weighted_A, weighted_b, rcond=None)[0]
    residual = A @ x - b
    residual_norm = np.linalg.norm(residual, ord=2)
    return x, residual_norm

# 示例用法
def test2():
    A = np.array([[1, 2], [3, 4], [5, 6]])
    b = np.array([1, 2, 3])
    G = np.array([[0.5, 0], [0, 2.0]])
    x, residual_norm = weighted_least_squares_column_weight(A, b, G)
    print("解向量 x:", x)
    print("残差范数:", residual_norm)

def ridge_regression(A: np.ndarray, b: np.ndarray, lambda_value: float) -> Tuple[np.ndarray, float]:
    """
    岭回归的正则化求解

    Args:
    - A (np.ndarray): 系数矩阵 A，形状为 (m, n)
    - b (np.ndarray): 右侧向量 b，形状为 (m,)
    - lambda_value (float): 正则化参数 lambda

    Returns:
    - x (np.ndarray): 解向量 x，形状为 (n,)
    - residual_norm (float): 残差范数 \|Ax - b\|_2
    """
    m, n = A.shape
    regularized_A = np.vstack((A, np.sqrt(lambda_value) * np.eye(n)))
    regularized_b = np.concatenate((b, np.zeros(n)))
    x = np.linalg.lstsq(regularized_A, regularized_b, rcond=None)[0]
    residual = A @ x - b
    residual_norm = np.linalg.norm(residual, ord=2)
    return x, residual_norm

# 示例用法
def test3():
    A = np.array([[1, 2], [3, 4]])
    b = np.array([1, 2])
    lambda_value = 0.1
    x, residual_norm = ridge_regression(A, b, lambda_value)
    print("解向量 x:", x)
    print("残差范数:", residual_norm)

def tikhonov_regularization(A: np.ndarray, b: np.ndarray, B: np.ndarray, lambda_value: float) -> Tuple[np.ndarray, float]:
    """
    Tikhonov正则化的求解

    Args:
    - A (np.ndarray): 系数矩阵 A，形状为 (m, n)
    - b (np.ndarray): 右侧向量 b，形状为 (m,)
    - B (np.ndarray): 正则化矩阵 B，形状为 (p, n)
    - lambda_value (float): 正则化参数 lambda

    Returns:
    - x (np.ndarray): 解向量 x，形状为 (n,)
    - residual_norm (float): 残差范数 \|Ax - b\|_2
    """
    m, n = A.shape
    p = B.shape[0]
    regularized_A = np.vstack((A, np.sqrt(lambda_value) * B))
    regularized_b = np.concatenate((b, np.zeros(p)))
    x = np.linalg.lstsq(regularized_A, regularized_b, rcond=None)[0]
    residual = A @ x - b
    residual_norm = np.linalg.norm(residual, ord=2)
    return x, residual_norm

# 示例用法
def test4():
    A = np.array([[1, 2], [3, 4]])
    b = np.array([1, 2])
    B = np.array([[1, 0], [0, 1]])
    lambda_value = 0.1
    x, residual_norm = tikhonov_regularization(A, b, B, lambda_value)
    print("解向量 x:", x)
    print("残差范数:", residual_norm)

if __name__ == "__main__":
    test1()
    test2()
    test3()
    test4()