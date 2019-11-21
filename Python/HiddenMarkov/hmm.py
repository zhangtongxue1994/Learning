# encoding=utf8

import numpy as np
import csv
from tqdm import tqdm


# reference: https://github.com/WenDesi/lihang_book_algorithm/tree/master/hmm
# np.seterr(divide='ignore',invalid='ignore')

class HMM(object):
    def __init__(self, N, M, init_array):
        self.A = init_array  # 状态转移概率矩阵
        self.B = init_array  # 观测概率矩阵
        self.Pi = np.array([1.0 / N] * N)  # 初始状态概率矩阵

        self.N = N  # 可能的状态数
        self.M = M  # 可能的观测数

    def cal_probality(self, O):
        self.T = len(O)
        self.O = O

        self.forward()
        return sum(self.alpha[self.T - 1])

    def forward(self):
        """
        前向算法
        """
        self.alpha = np.zeros((self.T, self.N))

        # 公式 10.15
        for i in range(self.N):
            self.alpha[0][i] = self.Pi[i] * self.B[i][self.O[0]]

        # 公式10.16
        for t in range(1, self.T):
            for i in range(self.N):
                sum = 0
                for j in range(self.N):
                    sum += self.alpha[t - 1][j] * self.A[j][i]
                self.alpha[t][i] = sum * self.B[i][self.O[t]]

    def backward(self):
        """
        后向算法
        """
        self.beta = np.zeros((self.T, self.N))

        # 公式10.19
        for i in range(self.N):
            self.beta[self.T - 1][i] = 1

        # 公式10.20
        for t in range(self.T - 2, -1, -1):
            for i in range(self.N):
                for j in range(self.N):
                    self.beta[t][i] += self.A[i][j] * self.B[j][self.O[t + 1]] * self.beta[t + 1][j]

    def cal_gamma(self, i, t):
        """
        公式 10.24
        """
        numerator = self.alpha[t][i] * self.beta[t][i]
        denominator = 0

        for j in range(self.N):
            denominator += self.alpha[t][j] * self.beta[t][j]

        return numerator / denominator

    def cal_ksi(self, i, j, t):
        """
        公式 10.26
        """

        numerator = self.alpha[t][i] * self.A[i][j] * self.B[j][self.O[t + 1]] * self.beta[t + 1][j]
        denominator = 0

        for i in range(self.N):
            for j in range(self.N):
                denominator += self.alpha[t][i] * self.A[i][j] * self.B[j][self.O[t + 1]] * self.beta[t + 1][j]

        return numerator / denominator

    def train(self, O, MaxSteps=100):
        self.T = len(O)
        self.O = O

        # 递推
        for step in tqdm(range(MaxSteps)):
            # print('step:', step)
            tmp_A = np.zeros((self.N, self.N))
            tmp_B = np.zeros((self.N, self.M))
            tmp_pi = np.array([0.0] * self.N)

            self.forward()
            self.backward()

            # a_{ij}
            for i in range(self.N):
                for j in range(self.N):
                    numerator = 0.0
                    denominator = 0.0 + 0.00000001
                    for t in range(self.T - 1):
                        numerator += self.cal_ksi(i, j, t)
                        denominator += self.cal_gamma(i, t)
                    tmp_A[i][j] = numerator / denominator

            # b_{jk}
            for j in range(self.N):
                for k in range(self.M):
                    numerator = 0.0
                    denominator = 0.0 + 0.00000001
                    for t in range(self.T):
                        if k == self.O[t]:
                            numerator += self.cal_gamma(j, t)
                        denominator += self.cal_gamma(j, t)
                    tmp_B[j][k] = numerator / denominator

            # pi_i
            for i in range(self.N):
                tmp_pi[i] = self.cal_gamma(i, 0)

            self.A = tmp_A
            self.B = tmp_B
            self.Pi = tmp_pi

    def generate(self, length):
        import random
        I = []

        # start
        ran = random.randint(0, 1000) / 1000.0
        i = 0
        while self.Pi[i] < ran or self.Pi[i] < 0.0001:
            ran -= self.Pi[i]
            i += 1
        I.append(i)

        # 生成状态序列
        for i in range(1, length):
            last = I[-1]
            ran = random.randint(0, 1000) / 1000.0
            i = 0
            while self.A[last][i] < ran or self.A[last][i] < 0.0001:
                ran -= self.A[last][i]
                i += 1
            I.append(i)

        # 生成观测序列
        Y = []
        for i in range(length):
            k = 0
            ran = random.randint(0, 1000) / 1000.0
            while self.B[I[i]][k] < ran or self.B[I[i]][k] < 0.0001:
                ran -= self.B[I[i]][k]
                k += 1
            Y.append(k)

        return Y


if __name__ == '__main__':
    # excel AB矩阵
    temp = np.array([[20, 1, 7, 2, 0, 2, 2, 2, 2, 5, 3, 3, 2],
                     [1, 20, 1, 0, 1, 4, 4, 0, 0, 0, 3, 3, 3],
                     [7, 1, 20, 0, 0, 1, 1, 2, 0, 7, 2, 2, 1],
                     [2, 0, 0, 20, 0, 2, 0, 0, 0, 2, 0, 0, 2],
                     [0, 1, 0, 0, 20, 1, 1, 1, 13, 1, 0, 0, 1],
                     [2, 4, 1, 2, 1, 20, 3, 0, 2, 0, 3, 3, 4],
                     [2, 4, 1, 0, 1, 3, 20, 0, 1, 0, 7, 9, 4],
                     [2, 0, 2, 0, 1, 0, 0, 20, 1, 5, 0, 0, 2],
                     [2, 0, 0, 0, 13, 2, 1, 1, 20, 1, 1, 2, 1],
                     [5, 0, 7, 2, 1, 0, 0, 5, 1, 20, 1, 1, 2],
                     [3, 3, 2, 0, 0, 3, 7, 0, 1, 1, 20, 6, 1],
                     [3, 3, 2, 0, 0, 3, 9, 0, 2, 1, 6, 20, 3],
                     [2, 3, 1, 2, 1, 4, 4, 2, 1, 2, 1, 3, 20]], dtype=float)
    for i in range(temp.shape[0]):
        temp[i, :] = temp[i, :] / sum(temp[i, :])

    # 建立模型，13个状态，13个观测
    hmm = HMM(N=13, M=13, init_array=temp)
    O = [6, 4, 4, 6, 9, 8, 7, 11, 0, 1]  # 主题数-1等于观测的状态
    hmm.train(O, MaxSteps=10)

    # 得到最终参数
    C = hmm.A
    D = hmm.B
    Pi = hmm.Pi

    print('C:', C)

    print('D:', D)

    # 后三年的预测

    pred = hmm.generate(3)
    # print('预测后三年的结果（+1等于topic）:', pred)

    # 2018年的数据
    num_2018 = [0.14077, 0.121775, 0.113685, 0.104277, 0.09469, 0.09248, 0.069601, 0.066073, 0.057769, 0.056557,
                0.039095, 0.026158, 0.017071]

    num_2019 = []
    for i in range(13):
        num_2019.append(sum(num_2018 * C[:, i]))
    print('num_2019:', num_2019 / sum(num_2019))

    num_2020 = []
    for i in range(13):
        num_2020.append(sum(num_2019 * C[:, i]))
    print('num_2020:', num_2020 / sum(num_2020))

    num_2021 = []
    for i in range(13):
        num_2021.append(sum(num_2020 * C[:, i]))
    print('num_2021:', num_2021 / sum(num_2021))
