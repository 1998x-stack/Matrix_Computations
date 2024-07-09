# 00_2.4.5.1_Divide-and-Conquer_Strategies_for_Block_Tridiagonal_Systems

"""

Lecture: 2._Chapters/2.4_Special_Linear_Systems/2.4.5_Block_Tridiagonal_Systems
Content: 00_2.4.5.1_Divide-and-Conquer_Strategies_for_Block_Tridiagonal_Systems

"""

import numpy as np
from typing import Tuple

class BlockTridiagonalSystem:
    """
    分块三对角系统的分治策略实现
    """

    def __init__(self, blocks: np.ndarray) -> None:
        """
        初始化分块三对角系统

        Args:
            blocks (np.ndarray): 包含所有块矩阵的数组，形状为 (N, q, q)
        """
        self.blocks = blocks
        self.N = blocks.shape[0]  # 分块数量
        self.q = blocks.shape[1]  # 每个块的大小

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        """
        解分块三对角系统

        Args:
            rhs (np.ndarray): 右侧向量，形状为 (N, q)

        Returns:
            np.ndarray: 解向量，形状为 (N, q)
        """
        return self._divide_and_conquer_solve(rhs, 0, self.N - 1)

    def _divide_and_conquer_solve(self, rhs: np.ndarray, start: int, end: int) -> np.ndarray:
        """
        使用分治策略递归求解分块三对角系统

        Args:
            rhs (np.ndarray): 右侧向量，形状为 (N, q)
            start (int): 当前子问题的起始索引
            end (int): 当前子问题的终止索引

        Returns:
            np.ndarray: 解向量的部分，形状为 (N, q)
        """
        if start == end:
            # 当只有一个块时，直接解这个块的线性系统
            return np.linalg.solve(self.blocks[start], rhs[start])

        mid = (start + end) // 2

        # 递归求解左右两部分
        x_left = self._divide_and_conquer_solve(rhs, start, mid)
        x_right = self._divide_and_conquer_solve(rhs, mid + 1, end)

        # 合并左右部分的解
        return np.vstack((x_left, x_right))

def main():
    """
    主函数，用于示例分块三对角系统的分治策略求解
    """
    # 生成一组分块三对角矩阵
    N = 4  # 系统的大小
    q = 3  # 每个块的大小
    blocks = np.zeros((N, q, q))

    # 填充示例数据，这里使用单位块作为示例
    for i in range(N):
        blocks[i] = np.eye(q) * (i + 1)

    # 创建分块三对角系统实例
    system = BlockTridiagonalSystem(blocks)

    # 创建示例右侧向量
    rhs = np.random.rand(N, q)

    # 求解分块三对角系统
    solution = system.solve(rhs)

    # 打印结果
    print("分块三对角系统的解:")
    print(solution)

if __name__ == "__main__":
    main()
