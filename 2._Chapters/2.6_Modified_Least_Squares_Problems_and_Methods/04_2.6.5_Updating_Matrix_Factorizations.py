# 04_2.6.5_Updating_Matrix_Factorizations

"""

Lecture: 2._Chapters/2.6_Modified_Least_Squares_Problems_and_Methods
Content: 04_2.6.5_Updating_Matrix_Factorizations

"""

import numpy as np
from typing import Tuple

class MatrixFactorizationUpdater:
    """
    矩阵分解更新类，包括QR分解和Cholesky分解的更新操作
    """

    @staticmethod
    def update_qr_add_row(Q: np.ndarray, R: np.ndarray, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        更新QR分解，添加一行

        Args:
            Q (np.ndarray): 原QR分解中的Q矩阵，形状为 (m, n)
            R (np.ndarray): 原QR分解中的R矩阵，形状为 (n, n)
            u (np.ndarray): 要添加的新行，形状为 (n,)

        Returns:
            Tuple[np.ndarray, np.ndarray]: 更新后的Q和R矩阵
        """
        m, n = Q.shape
        u_hat = np.dot(Q.T, u)
        u_residual = u - np.dot(Q, u_hat)
        norm_residual = np.linalg.norm(u_residual)
        if norm_residual > 1e-10:
            q_new = u_residual / norm_residual
            Q_new = np.column_stack((Q, q_new))
            R_new = np.vstack((np.column_stack((R, u_hat)), np.append(np.zeros(n), norm_residual)))
        else:
            Q_new = Q
            R_new = R
        return Q_new, R_new

    @staticmethod
    def downdate_qr_remove_row(Q: np.ndarray, R: np.ndarray, row_index: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        更新QR分解，删除一行

        Args:
            Q (np.ndarray): 原QR分解中的Q矩阵，形状为 (m, n)
            R (np.ndarray): 原QR分解中的R矩阵，形状为 (n, n)
            row_index (int): 要删除的行索引

        Returns:
            Tuple[np.ndarray, np.ndarray]: 更新后的Q和R矩阵
        """
        m, n = Q.shape
        Q = np.delete(Q, row_index, axis=0)
        R = np.delete(R, row_index, axis=0)
        for i in range(row_index, m-1):
            G = np.eye(n)
            a, b = R[i, i], R[i+1, i]
            r = np.hypot(a, b)
            c, s = a / r, -b / r
            G[i:i+2, i:i+2] = [[c, -s], [s, c]]
            R = np.dot(G, R)
            Q[:, i:i+2] = np.dot(Q[:, i:i+2], G.T)
        return Q, R

    @staticmethod
    def update_cholesky(L: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        更新Cholesky分解，添加行

        Args:
            L (np.ndarray): 原Cholesky分解中的L矩阵，形状为 (n, n)
            u (np.ndarray): 要添加的行，形状为 (n,)

        Returns:
            np.ndarray: 更新后的L矩阵
        """
        n = L.shape[0]
        u_new = np.append(u, np.linalg.norm(u))
        L_new = np.vstack((np.hstack((L, np.zeros((n, 1)))), u_new))
        return L_new

    @staticmethod
    def downdate_cholesky(L: np.ndarray, row_index: int) -> np.ndarray:
        """
        更新Cholesky分解，删除行

        Args:
            L (np.ndarray): 原Cholesky分解中的L矩阵，形状为 (n, n)
            row_index (int): 要删除的行索引

        Returns:
            np.ndarray: 更新后的L矩阵
        """
        L = np.delete(L, row_index, axis=0)
        for i in range(row_index, L.shape[0]):
            G = np.eye(L.shape[0])
            a, b = L[i, i], L[i+1, i]
            r = np.hypot(a, b)
            c, s = a / r, -b / r
            G[i:i+2, i:i+2] = [[c, -s], [s, c]]
            L = np.dot(G, L)
        return L

def main():
    """
    主函数，用于示例矩阵分解的更新操作
    """
    # 示例QR分解的更新
    A = np.array([
        [12, -51, 4],
        [6, 167, -68],
        [-4, 24, -41]
    ], dtype=float)
    Q, R = np.linalg.qr(A)
    u = np.array([1, 2, 3], dtype=float)

    print("原始矩阵 A 的 QR 分解:")
    print("Q:\n", Q)
    print("R:\n", R)

    Q_new, R_new = MatrixFactorizationUpdater.update_qr_add_row(Q, R, u)
    print("\n添加一行后的 QR 分解:")
    print("Q:\n", Q_new)
    print("R:\n", R_new)

    Q_downdated, R_downdated = MatrixFactorizationUpdater.downdate_qr_remove_row(Q, R, 1)
    print("\n删除一行后的 QR 分解:")
    print("Q:\n", Q_downdated)
    print("R:\n", R_downdated)

    # 示例Cholesky分解的更新
    A = np.array([
        [4, 2],
        [2, 2]
    ], dtype=float)
    L = np.linalg.cholesky(A)
    u = np.array([1, 1], dtype=float)

    print("\n原始矩阵 A 的 Cholesky 分解:")
    print("L:\n", L)

    L_new = MatrixFactorizationUpdater.update_cholesky(L, u)
    print("\n添加一行后的 Cholesky 分解:")
    print("L:\n", L_new)

    L_downdated = MatrixFactorizationUpdater.downdate_cholesky(L, 1)
    print("\n删除一行后的 Cholesky 分解:")
    print("L:\n", L_downdated)

if __name__ == "__main__":
    main()
