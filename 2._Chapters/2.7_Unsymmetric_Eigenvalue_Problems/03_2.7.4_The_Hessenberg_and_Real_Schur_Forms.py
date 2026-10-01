# 03_2.7.4_The_Hessenberg_and_Real_Schur_Forms

"""

Lecture: 2._Chapters/2.7_Unsymmetric_Eigenvalue_Problems
Content: 03_2.7.4_The_Hessenberg_and_Real_Schur_Forms

"""

import numpy as np
from typing import Tuple

def householder_transform(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    将矩阵转换为Hessenberg形式的Householder变换。

    Args:
        A (np.ndarray): 输入矩阵。

    Returns:
        H (np.ndarray): Hessenberg形式的矩阵。
        Q (np.ndarray): 正交矩阵。
    """
    m, n = A.shape
    Q = np.eye(m)
    H = A.copy()

    for i in range(n-2):
        x = H[i+1:, i]
        e = np.zeros_like(x)
        e[0] = np.linalg.norm(x) * (1 if x[0] == 0 else np.sign(x[0]))
        u = x + e
        u = u / np.linalg.norm(u)

        H[i+1:, i:] -= 2 * np.outer(u, u @ H[i+1:, i:])
        H[:, i+1:] -= 2 * np.outer(H[:, i+1:] @ u, u)
        Q[i+1:] -= 2 * np.outer(u, u @ Q[i+1:])

    return H, Q.T


def qr_iteration(H: np.ndarray, max_iter: int = 1000, tol: float = 1e-10) -> Tuple[np.ndarray, np.ndarray]:
    """
    对Hessenberg矩阵进行QR迭代，逼近实Schur形式。

    Args:
        H (np.ndarray): Hessenberg矩阵。
        max_iter (int): 最大迭代次数。
        tol (float): 收敛容差。

    Returns:
        T (np.ndarray): 上三角块状矩阵。
        Q (np.ndarray): 正交矩阵。
    """
    n = H.shape[0]
    Q_total = np.eye(n)

    for _ in range(max_iter):
        Q, R = np.linalg.qr(H)
        H = R @ Q
        Q_total = Q_total @ Q

        # 检查收敛性
        off_diagonal_norm = np.sum(np.abs(H[np.tril_indices(n, -1)]))
        if off_diagonal_norm < tol:
            break

    return H, Q_total


class SchurForm:
    """
    将矩阵转换为实Schur形式的类。

    Attributes:
        A (np.ndarray): 输入矩阵。
    """

    def __init__(self, A: np.ndarray):
        """
        初始化SchurForm类。

        Args:
            A (np.ndarray): 输入矩阵。
        """
        self.A = A

    def compute_schur_form(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算矩阵的实Schur形式。

        Returns:
            T (np.ndarray): 实Schur形式的上三角块状矩阵。
            Q (np.ndarray): 实Schur形式的正交矩阵。
        """
        # 转换为Hessenberg形式
        H, Q = householder_transform(self.A)

        # QR迭代
        T, Q_schur = qr_iteration(H)

        # 最终的正交矩阵
        Q_final = Q @ Q_schur

        return T, Q_final


def main():
    """
    主函数，用于测试SchurForm类。
    """
    A = np.array([[4, 1, 2],
                  [3, 4, 1],
                  [1, 1, 3]], dtype=float)

    schur = SchurForm(A)
    T, Q = schur.compute_schur_form()

    print("原始矩阵 A:")
    print(A)
    print("\nSchur形式的上三角块状矩阵 T:")
    print(T)
    print("\n正交矩阵 Q:")
    print(Q)
    print("\n验证 A = Q T Q^T:")
    print(np.allclose(A, Q @ T @ Q.T))

if __name__ == "__main__":
    main()
