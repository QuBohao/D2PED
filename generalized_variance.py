#--coding:utf-8--
import numpy as np

def compute_generalized_variance(pop):
    cov_matrix = np.cov(pop)       # 计算各个agent的embedding之间的协方差矩阵
    return np.linalg.det(cov_matrix)