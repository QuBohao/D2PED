# --coding:utf-8--
import itertools
import random
import numpy as np
import torch

import ray
from collections import namedtuple

from gv.generalized_variance import compute_generalized_variance


# Transition = namedtuple('Transition', ('state_action', 'flag'))


# @ray.remote，表明该类可以被ray远程操作
@ray.remote
class ReplayBuffer:

    def __init__(self, policy_number, transformer):
        self.size = 0
        self.episode = 0
        self.policy_number = policy_number

        self.transformer = transformer

        self.transformer_training = False

        self.memory = []

        for i in range(policy_number):
            self.memory.append([])

    def push(self, flag, transition):
        self.memory[flag].append(transition)

    def get_size(self):
        temp = 0
        for i in range(self.policy_number):
            temp += len(self.memory[i])
        self.size = temp
        return self.size

    # def count_size(self, count):
    #     self.size = self.size + count

    def buffer_reset(self):
        self.size = 0
        self.memory.clear()

    def extract_data(self):
        return [list(self.memory[i]) for i in range(self.policy_number)]

    #     return Transition(*zip(*self.get_all()))

    def get_transformer_training_bool(self):
        return self.transformer_training

    def change_transformer_training_bool(self, training_flag):
        self.transformer_training = training_flag

    def get_all(self, flag):
        return self.get(0, len(self.memory[flag]))

    def get(self, flag, start_idx: int, end_idx: int):
        transitions = list(itertools.islice(self.memory[flag], start_idx, end_idx))
        return transitions

    def update_transformer(self, model):
        self.transformer = model

    # def count_episode(self):
    #     self.episode += 1
    #
    # def get_total_episode(self):
    #     return self.episode

    # 在replaybuffer中为n个agent计算共同的GV
    def get_generalized_variance(self):
        a = np.array([])

        # todo 该循环调试使用
        for i in range(self.policy_number):
            self.memory[i] = self.memory[0]

        for i in range(self.policy_number):
            a = np.append(a, len(self.memory[i]))

        if np.count_nonzero(a) < self.policy_number:
            print("GV return 0")
            return 0

        index_range = int(a[np.argsort(a)[0]])
        index = random.randint(0, index_range)

        matrix = torch.FloatTensor([list(self.memory[i][index][:-1]) for i in range(self.policy_number)])
        # todo memory过transformer
        pop = self.transformer(matrix)

        # pop = np.array([list(self.memory[i][index][:-1]) for i in range(self.policy_number)])
        return compute_generalized_variance(pop)
