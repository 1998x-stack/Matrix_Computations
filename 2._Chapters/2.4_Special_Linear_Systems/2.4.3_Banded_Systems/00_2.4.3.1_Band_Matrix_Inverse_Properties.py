# 00_2.4.3.1_Band_Matrix_Inverse_Properties

"""

Lecture: 2._Chapters/2.4_Special_Linear_Systems/2.4.3_Banded_Systems
Content: 00_2.4.3.1_Band_Matrix_Inverse_Properties

"""

import numpy as np
from scipy.linalg import lu_factor, lu_solve
from typing import Tuple

class BandMatrix:
    """
    带状矩阵及其逆矩阵的性质
    """

    def __init__(self, matrix: np.ndarray, lower_bandwidth: int, upper_bandwidth: int) -> None:
        """
        初始化带状矩阵

        Args:
            matrix (np.ndarray): 带状矩阵
            lower_bandwidth (int): 矩阵的下带宽
            upper_bandwidth (int): 矩阵的上带宽
        """
        self.matrix = matrix
        self.lower_bandwidth = lower_bandwidth
        self.upper_bandwidth = upper_bandwidth

    def inverse(self) -> np.ndarray:
        """
        计算带状矩阵的逆矩阵

        Returns:
            np.ndarray: 带状矩阵的逆矩阵
        """
        n = self.matrix.shape[0]
        inv_matrix = np.zeros_like(self.matrix)

        # LU分解
        lu, piv = lu_factor(self.matrix)
        
        # 计算单位矩阵的每一列的逆
        for i in range(n):
            e_i = np.zeros(n)
            e_i[i] = 1.0
            inv_matrix[:, i] = lu_solve((lu, piv), e_i)

        return inv_matrix

def main():
    """
    主函数，用于示例带状矩阵及其逆矩阵的性质
    """
    # 生成一个带状矩阵
    A = np.array([
        [4, 1, 0, 0],
        [1, 4, 1, 0],
        [0, 1, 4, 1],
        [0, 0, 1, 4]
    ])

    lower_bandwidth = 1
    upper_bandwidth = 1

    # 初始化带状矩阵
    band_matrix = BandMatrix(A, lower_bandwidth, upper_bandwidth)
    
    # 计算带状矩阵的逆矩阵
    inv_A = band_matrix.inverse()
    
    # 打印结果
    print("带状矩阵A:")
    print(A)
    print("\n带状矩阵A的逆矩阵:")
    print(inv_A)
    print("\n验证A @ A^-1是否等于单位矩阵:")
    print(np.allclose(np.dot(A, inv_A), np.eye(A.shape[0])))

if __name__ == "__main__":
    main()