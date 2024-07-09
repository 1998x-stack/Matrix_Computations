# 06_2.4.7_Classical_Methods_for_Toeplitz_Systems

"""

Lecture: 2._Chapters/2.4_Special_Linear_Systems
Content: 06_2.4.7_Classical_Methods_for_Toeplitz_Systems

"""

import numpy as np
from typing import Tuple

class ToeplitzSolver:
    """
    使用经典方法求解Toeplitz系统的类
    """

    def __init__(self, first_column: np.ndarray, first_row: np.ndarray) -> None:
        """
        初始化Toeplitz矩阵的第一列和第一行

        Args:
            first_column (np.ndarray): Toeplitz矩阵的第一列
            first_row (np.ndarray): Toeplitz矩阵的第一行
        """
        self.first_column = first_column
        self.first_row = first_row
        self.n = len(first_column)
        self.toeplitz_matrix = self._construct_toeplitz_matrix()

    def _construct_toeplitz_matrix(self) -> np.ndarray:
        """
        构建Toeplitz矩阵

        Returns:
            np.ndarray: 构建的Toeplitz矩阵
        """
        T = np.zeros((self.n, self.n))
        for i in range(self.n):
            T[i, :] = np.roll(self.first_row, i)
            T[i, :i] = self.first_column[1:i+1][::-1]
        return T

    def solve_system(self, b: np.ndarray) -> np.ndarray:
        """
        使用Levinson-Durbin算法求解Toeplitz系统

        Args:
            b (np.ndarray): 系统的右端项

        Returns:
            np.ndarray: 系统的解
        """
        n = len(b)
        x = np.zeros(n)
        y = np.zeros(n)
        g = np.zeros(n)
        h = np.zeros(n)

        # 初始条件
        x[0] = b[0] / self.first_column[0]
        y[0] = self.first_column[0]
        g[0] = b[0] / self.first_column[0]
        h[0] = self.first_column[0]

        for k in range(1, n):
            # 计算alpha和beta
            alpha = -sum(self.first_column[1:k+1] * y[k-1::-1])
            beta = sum(self.first_column[1:k+1] * g[k-1::-1])
            
            y[k] = alpha / (1 - beta)
            g[k] = (b[k] - sum(self.first_column[1:k+1] * x[k-1::-1])) / (self.first_column[0] - beta)
            
            for j in range(k):
                x[j] -= g[k] * y[k-j-1]
            
            x[k] = g[k]
        
        return x

def main():
    """
    主函数，用于示例Toeplitz系统的求解
    """
    # 定义Toeplitz矩阵的第一列和第一行
    first_column = np.array([4, 1, 2, 3])
    first_row = np.array([4, 5, 6, 7])
    
    # 定义右端项
    b = np.array([1, 2, 3, 4])
    
    # 初始化Toeplitz求解器
    solver = ToeplitzSolver(first_column, first_row)
    
    # 打印构建的Toeplitz矩阵
    print("构建的Toeplitz矩阵:")
    print(solver.toeplitz_matrix)
    
    # 求解Toeplitz系统
    solution = solver.solve_system(b)
    
    # 打印求解结果
    print("系统的解:")
    print(solution)

if __name__ == "__main__":
    main()