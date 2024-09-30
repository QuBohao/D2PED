# --coding:utf-8--
import os
import random
import time

import ray
from argparse import ArgumentParser
import numpy as np
import torch
import gym
from gym.wrappers import RescaleAction
import torch.nn as nn
import torch.optim as optim

from policy.td3 import TD3_Agent
from policy.train_agent import train_agent_model_free
from replay_buffer import ReplayBuffer
from transformer.transformer import ViT
from tqdm.notebook import tqdm
from torch.utils.data import DataLoader, Dataset


# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def wait_and_discriminator(buffer, buffer_size):
    print('wait sample data...')
    begin = time.time()
    # 临时方案
    while ray.get(buffer.get_size.remote()) < buffer_size:
        print('replay_buffer_size:{}'.format(ray.get(buffer.get_size.remote())))
        time.sleep(6)
    print('replay_buffer_size:{}'.format(ray.get(buffer.get_size.remote())))
    print('policy training process time:{}'.format(str(time.time() - begin)))


class StatesActionsDataset(Dataset):
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        self.filelength = len(self.file_list)
        return self.filelength

    def __getitem__(self, idx):
        trajectory = self.file_list[idx]
        label = int(trajectory[-1])
        trajectory = trajectory[:-1]

        return trajectory, label


def transformer_train_local(model, replay_buffer, params, criterion, optimizer, device):

    model.to(device)
    epochs = 20
    global total_update
    start_time = time.time()
    replay_buffer.change_transformer_training_bool.remote(True)

    data_list = ray.get(replay_buffer.extract_data.remote())
    train_list = []
    for i in range(params['policy_number']):
        train_list += data_list[i]

    train_data = StatesActionsDataset(train_list)
    train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=True)

    for epoch in range(epochs):
        epoch_loss = 0
        epoch_accuracy = 0

        for data, label in tqdm(train_loader):
            data = data.to(device)
            label = label.to(device)

            output = model(data)
            loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc = (output.argmax(dim=1) == label).float().mean()
            epoch_accuracy += acc / len(train_loader)
            epoch_loss += loss / len(train_loader)


        print(f"Epoch : {epoch + 1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} ")

    total_update += 1

    replay_buffer.update_transformer.remote(model)
    replay_buffer.change_transformer_training_bool.remote(False)
    print('transformer_update time:{}'.format(total_update))
    # print('update time:{}'.format(agents[0].update_time))
    print('current local_train process duration: {}'.format(str(time.time() - start_time)))
    # return agents[0].update_time


def main():
    parser = ArgumentParser()
    parser.add_argument('--env', type=str, default='Humanoid-v2')
    parser.add_argument('--seed', type=int, default=100)
    parser.add_argument('--use_obs_filter', dest='obs_filter', action='store_true')
    parser.add_argument('--policy_number', type=int, default=5)
    parser.add_argument('--update_every_n_steps', type=int, default=1)
    parser.add_argument('--n_random_actions', type=int, default=25000)
    parser.add_argument('--n_collect_steps', type=int, default=1000)

    parser.add_argument('--buffer_size', type=int, default=50) #20000
    parser.add_argument('--traj_step', type=int, default=400) #20000

    args = parser.parse_args()
    params = vars(args)

    env_name = params['env']

    env = gym.make(env_name)
    env = RescaleAction(env, -1, 1)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    seed = params['seed']

    seed_everything(seed)

    policy_number = params['policy_number']

    policies = [TD3_Agent(seed, state_dim, action_dim) for i in range(policy_number)]

    transformer = ViT(
        traj_size=(state_dim+action_dim) * params['traj_step'],
        patch_size=state_dim+action_dim,
        num_classes=policy_number,
        dim=400,
        depth=6,
        heads=16,
        mlp_dim=1024,
        dropout=0.1,
        emb_dropout=0.1
    )

    # transformer loss function
    criterion = nn.CrossEntropyLoss()
    # transformer optimizer
    optimizer = optim.Adam(transformer.parameters(), lr=3e-5)

    replay_buffer = ReplayBuffer.remote(policy_number, transformer)

    print('transformer device: {}'.format(device))

    # 为了让replay_buffer建造完成
    time.sleep(1)

    _ = [train_agent_model_free.remote(agent=policies[i], env_name=env_name, params=params, common_replay_buffer=replay_buffer, flag=i) for
         i in range(policy_number)]



    while True:
        wait_and_discriminator(replay_buffer, params['buffer_size'])
        __ = transformer_train_local(transformer, replay_buffer, params, criterion, optimizer, device)

        if args.schedule_adam == 'linear':
            if (update_time + 1) % 200 == 0:
                # ep_ratio = 1 - (current_episode / args.delay_episode)
                ep_ratio = 0.95
                lr_now = lr_now * ep_ratio
                # set learning rate
                # ref: https://stackoverflow.com/questions/48324152/
                for j in range(args.fighter_num):
                    for g in optimizers[j].param_groups:
                        g['lr'] = lr_now


if __name__ == '__main__':
    main()
