# 07_2.4.8_Circulant_and_Discrete_Poisson_Systems

"""

Lecture: 2._Chapters/2.4_Special_Linear_Systems
Content: 07_2.4.8_Circulant_and_Discrete_Poisson_Systems

"""

import numpy as np
from typing import Tuple

class CirculantPoissonSolver:
    """
    循环矩阵和离散Poisson系统求解器
    """

    def __init__(self, f: np.ndarray, boundary_conditions: Tuple[float, float]) -> None:
        """
        初始化求解器

        Args:
            f (np.ndarray): 离散Poisson方程的右端项
            boundary_conditions (Tuple[float, float]): 边界条件 (u(a), u(b))
        """
        self.f = f
        self.n = len(f)
        self.u_a, self.u_b = boundary_conditions
        self.solution = None

    def solve_system(self) -> np.ndarray:
        """
        使用快速傅里叶变换（FFT）求解离散Poisson方程

        Returns:
            np.ndarray: 求解得到的离散Poisson方程的解向量
        """
        # 构造右端项
        b = self._construct_rhs()

        # 构造循环矩阵的第一列
        first_column = self._construct_first_column()

        # 使用FFT求解循环矩阵系统
        x = np.fft.ifft(np.fft.fft(b) / np.fft.fft(first_column)).real

        # 应用边界条件
        x += self.u_a + (self.u_b - self.u_a) * np.arange(self.n) / (self.n - 1)

        self.solution = x
        return x

    def _construct_rhs(self) -> np.ndarray:
        """
        构造离散Poisson方程的右端项

        Returns:
            np.ndarray: 离散Poisson方程的右端项向量
        """
        return -self.f

    def _construct_first_column(self) -> np.ndarray:
        """
        构造循环矩阵的第一列

        Returns:
            np.ndarray: 循环矩阵的第一列向量
        """
        first_column = np.zeros(self.n)
        first_column[0] = 2.0
        first_column[1] = -1.0
        first_column[self.n - 1] = -1.0
        return first_column

def main():
    """
    主函数，用于示例循环矩阵和离散Poisson系统的求解
    """
    # 定义离散Poisson方程的右端项和边界条件
    f = np.array([1.0, 2.0, 3.0, 4.0])
    boundary_conditions = (0.0, 0.0)  # 边界条件 (u(a), u(b))

    # 初始化求解器
    solver = CirculantPoissonSolver(f, boundary_conditions)
    
    # 求解离散Poisson方程
    solution = solver.solve_system()
    
    # 打印求解结果
    print("离散Poisson方程的解:")
    print(solution)

if __name__ == "__main__":
    main()