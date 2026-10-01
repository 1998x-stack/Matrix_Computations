# 00_2.8.7.1_Structured_Quadratic_Eigenvalue_Problem

"""

Lecture: 2._Chapters/2.8_Symmetric_Eigenvalue_Problems/2.8.7_Generalized_Eigenvalue_Problems_with_Symmetry
Content: 00_2.8.7.1_Structured_Quadratic_Eigenvalue_Problem

"""

import numpy as np
from scipy.linalg import eig

class StructuredQEPSolver:
    def __init__(self, M: np.ndarray, C: np.ndarray, K: np.ndarray):
        """
        初始化结构化二次特征值问题求解器。

        Args:
        - M (np.ndarray): 质量矩阵。
        - C (np.ndarray): 阻尼矩阵。
        - K (np.ndarray): 刚度矩阵。
        """
        assert M.shape == C.shape == K.shape, "所有输入矩阵必须具有相同的维度。"
        self.M = M
        self.C = C
        self.K = K
        self.n = M.shape[0]

    def linearize(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        将二次特征值问题线性化为广义特征值问题。

        Returns:
        - A (np.ndarray): 广义特征值问题中的矩阵 A。
        - B (np.ndarray): 广义特征值问题中的矩阵 B。
        """
        zero_matrix = np.zeros_like(self.M)
        identity_matrix = np.eye(self.n)

        A = np.block([
            [zero_matrix, identity_matrix],
            [-self.K, -self.C]
        ])
        
        B = np.block([
            [identity_matrix, zero_matrix],
            [zero_matrix, self.M]
        ])

        return A, B

    def solve(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        求解结构化二次特征值问题，返回特征值和特征向量。

        Returns:
        - eigenvalues (np.ndarray): 特征值。
        - eigenvectors (np.ndarray): 特征向量。
        """
        A, B = self.linearize()
        eigenvalues, eigenvectors = eig(A, B)
        return eigenvalues, eigenvectors

# 示例矩阵
M = np.array([
    [2, 0],
    [0, 1]
])

C = np.array([
    [0.1, 0.2],
    [0.2, 0.3]
])

K = np.array([
    [4, 1],
    [1, 3]
])

solver = StructuredQEPSolver(M, C, K)
eigenvalues, eigenvectors = solver.solve()

print("Eigenvalues:\n", eigenvalues)
print("Eigenvectors:\n", eigenvectors)
