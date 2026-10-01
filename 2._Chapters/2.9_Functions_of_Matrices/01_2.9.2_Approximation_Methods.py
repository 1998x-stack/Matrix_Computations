# 01_2.9.2_Approximation_Methods

"""

Lecture: 2._Chapters/2.9_Functions_of_Matrices
Content: 01_2.9.2_Approximation_Methods

"""

import numpy as np
from numpy.linalg import inv, norm
from typing import Callable, List

class MatrixFunctions:
    """
    用于近似矩阵函数的类，包括Jordan和Schur分析、Taylor近似、多项式评估以及Cauchy积分公式。
    """

    def __init__(self, A: np.ndarray):
        """
        初始化矩阵A。

        Args:
            A (np.ndarray): 需要计算函数的矩阵。
        """
        self.A = A
        self.n = A.shape[0]

    def jordan_approximation(self, f: Callable[[np.ndarray], np.ndarray], g: Callable[[np.ndarray], np.ndarray]) -> float:
        """
        使用Jordan分析方法近似计算矩阵函数。

        Args:
            f (Callable[[np.ndarray], np.ndarray]): 矩阵函数f(z)。
            g (Callable[[np.ndarray], np.ndarray]): 近似矩阵函数g(z)。

        Returns:
            float: 近似误差。
        """
        X, J = self.jordan_form(self.A)
        h = lambda z: f(z) - g(z)
        K2_X = norm(X) * norm(inv(X))
        error = K2_X * max([norm(h(Ji)) for Ji in J])
        return error

    def schur_approximation(self, f: Callable[[np.ndarray], np.ndarray], g: Callable[[np.ndarray], np.ndarray]) -> float:
        """
        使用Schur分析方法近似计算矩阵函数。

        Args:
            f (Callable[[np.ndarray], np.ndarray]): 矩阵函数f(z)。
            g (Callable[[np.ndarray], np.ndarray]): 近似矩阵函数g(z)。

        Returns:
            float: 近似误差。
        """
        Q, T = self.schur_form(self.A)
        N = np.triu(T, 1)
        omega = self.get_omega()
        r = min(self.n - 1, 10)
        h = lambda z: f(z) - g(z)
        error = sum((norm(N, 'fro')**i / np.math.factorial(i)) * max([abs(np.polyval(np.poly(h), z)) for z in omega]) for i in range(r))
        return error

    def taylor_approximation(self, f: Callable[[np.ndarray], np.ndarray], q: int) -> np.ndarray:
        """
        使用Taylor级数近似计算矩阵函数。

        Args:
            f (Callable[[np.ndarray], np.ndarray]): 矩阵函数f(z)。
            q (int): Taylor级数的阶数。

        Returns:
            np.ndarray: 近似的矩阵函数值。
        """
        c = [f(np.zeros((self.n, self.n)))]
        for k in range(1, q + 1):
            c.append(f(np.eye(self.n)))
        approx = sum(c[k] * np.linalg.matrix_power(self.A, k) for k in range(q + 1))
        return approx

    def polynomial_evaluation(self, coefficients: List[float]) -> np.ndarray:
        """
        使用Horner方法评估矩阵多项式。

        Args:
            coefficients (List[float]): 多项式系数。

        Returns:
            np.ndarray: 评估的多项式矩阵。
        """
        q = len(coefficients) - 1
        F = coefficients[q] * self.A + coefficients[q - 1] * np.eye(self.n)
        for k in range(q - 2, -1, -1):
            F = np.dot(self.A, F) + coefficients[k] * np.eye(self.n)
        return F

    def cauchy_integral(self, f: Callable[[complex], complex], gamma: List[complex]) -> np.ndarray:
        """
        使用Cauchy积分公式计算矩阵函数。

        Args:
            f (Callable[[complex], complex]): 复数函数f(z)。
            gamma (List[complex]): 积分路径上的点。

        Returns:
            np.ndarray: 近似的矩阵函数值。
        """
        integral = np.zeros((self.n, self.n), dtype=complex)
        I = np.eye(self.n)
        for z in gamma:
            integral += f(z) * inv(z * I - self.A)
        integral *= 1 / (2 * np.pi * 1j)
        return integral.real

    def jordan_form(self, A: np.ndarray) -> (np.ndarray, List[np.ndarray]):
        """
        计算矩阵的Jordan形式。

        Args:
            A (np.ndarray): 输入矩阵。

        Returns:
            (np.ndarray, List[np.ndarray]): Jordan形式的分解矩阵。
        """
        from scipy.linalg import jordan_form
        J, P = jordan_form(A)
        return P, [J]

    def schur_form(self, A: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        计算矩阵的Schur分解形式。

        Args:
            A (np.ndarray): 输入矩阵。

        Returns:
            (np.ndarray, np.ndarray): Schur分解形式的分解矩阵。
        """
        from scipy.linalg import schur
        T, Q = schur(A)
        return Q, T

    def get_omega(self) -> List[complex]:
        """
        获取包含矩阵特征值的闭合凸集。

        Returns:
            List[complex]: 闭合凸集上的点。
        """
        # 这里假设一个包含特征值的闭合凸集
        return [complex(1, 0), complex(-1, 0), complex(0, 1), complex(0, -1)]

# 示例使用
if __name__ == "__main__":
    # 定义矩阵A
    A = np.array([[4, 1], [2, 3]])

    # 创建MatrixFunctions实例
    matrix_funcs = MatrixFunctions(A)

    # 定义函数f(z)和g(z)
    f = np.exp
    g = lambda z: 1 + z + (z**2) / 2

    # 使用Jordan分析方法
    jordan_error = matrix_funcs.jordan_approximation(f, g)
    print(f"Jordan分析方法的近似误差: {jordan_error}")

    # 使用Schur分析方法
    schur_error = matrix_funcs.schur_approximation(f, g)
    print(f"Schur分析方法的近似误差: {schur_error}")

    # 使用Taylor级数近似
    taylor_approx = matrix_funcs.taylor_approximation(f, 10)
    print(f"Taylor级数近似的矩阵函数值:\n{taylor_approx}")

    # 使用Horner方法评估矩阵多项式
    coefficients = [1, -3, 2]  # 例如多项式z^2 - 3z + 2
    poly_eval = matrix_funcs.polynomial_evaluation(coefficients)
    print(f"Horner方法评估的多项式矩阵:\n{poly_eval}")

    # 使用Cauchy积分公式计算矩阵函数
    gamma = [complex(1, 1), complex(-1, 1), complex(-1, -1), complex(1, -1)]
    cauchy_approx = matrix_funcs.cauchy_integral(f, gamma)
    print(f"Cauchy积分公式计算的矩阵函数值:\n{cauchy_approx}")
