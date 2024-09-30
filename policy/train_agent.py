import random
import time
from argparse import ArgumentParser
from collections import deque, namedtuple

import gym
from gym.wrappers import RescaleAction
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from policy.td3 import TD3_Agent
from policy.utils import MeanStdevFilter, Transition, make_gif, make_checkpoint

import ray

# Transition2 = namedtuple('Transition', ('state_action', 'flag'))

@ray.remote
def train_agent_model_free(agent, env_name, params, common_replay_buffer, flag):

    
    update_timestep = params['update_every_n_steps']
    seed = params['seed']
    log_interval = 1000
    gif_interval = 500000
    n_random_actions = params['n_random_actions']
    # n_evals = params['n_evals']
    n_collect_steps = params['n_collect_steps']
    use_statefilter = params['obs_filter']
    # save_model = params['save_model']

    assert n_collect_steps > agent.batchsize, "We must initially collect as many steps as the batch size!"

    avg_length = 0
    time_step = 0
    cumulative_timestep = 0
    cumulative_log_timestep = 0
    n_updates = 0
    i_episode = 0
    log_episode = 0
    samples_number = 0
    episode_rewards = []
    episode_steps = []

    env = gym.make(env_name)
    env = RescaleAction(env, -1, 1)

    if use_statefilter:
        state_filter = MeanStdevFilter(env.env.observation_space.shape[0])
    else:
        state_filter = None

    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    env.seed(seed)
    env.action_space.np_random.seed(seed)

    max_steps = env.spec.max_episode_steps

    writer = SummaryWriter()

    local_buffer = np.array([])
    push_cumulative_timestep = 0

    while samples_number < 3e7:
        time_step = 0
        episode_reward = 0
        i_episode += 1
        log_episode += 1
        state = env.reset()
        if state_filter:
            state_filter.update(state)
        done = False

        count = 0
        while ray.get(common_replay_buffer.get_transformer_training_bool.remote()):
            time.sleep(60)
            count += 1
            print('policy, wait transformer training!')
            if count >= 600:
                print('ps dead, this process exit!')
                return None

        while (not done):

            # env.render()
            push_cumulative_timestep += 1

            cumulative_log_timestep += 1
            cumulative_timestep += 1
            time_step += 1
            samples_number += 1
            if samples_number < n_random_actions:
                action = env.action_space.sample()
            else:
                action = agent.get_action(state, state_filter=state_filter)
            nextstate, reward, done, _ = env.step(action)
            # if we hit the time-limit, it's not a 'real' done; we don't want to assign low value to those states
            real_done = False if time_step == max_steps else done
            agent.replay_pool.push(Transition(state, action, reward, nextstate, real_done))

            local_buffer = np.append(local_buffer, np.concatenate((state, action)))

            state = nextstate
            if state_filter:
                state_filter.update(state)
            episode_reward += reward
            if push_cumulative_timestep >= params['traj_step']:
                common_replay_buffer.push.remote(flag, np.append(local_buffer, flag))
                local_buffer = np.array([])
                push_cumulative_timestep = 0


            # update if it's time
            if cumulative_timestep % update_timestep == 0 and cumulative_timestep > n_collect_steps:
                GV = 0
                if n_updates > 1:
                    GV = common_replay_buffer.get_generalized_variance.remote()
                    # pass
                q1_loss, q2_loss, pi_loss = agent.optimize(update_timestep, GV ,state_filter=state_filter)
                n_updates += 1
            # # logging
            # if cumulative_timestep % log_interval == 0 and cumulative_timestep > n_collect_steps:
            #     writer.add_scalar('Loss/Q-func_1', q1_loss, n_updates)
            #     writer.add_scalar('Loss/Q-func_2', q2_loss, n_updates)
            #     #TODO: This may not work; fix this
            #     if pi_loss:
            #         writer.add_scalar('Loss/policy', pi_loss, n_updates)
            #     avg_length = np.mean(episode_steps)
            #     running_reward = np.mean(episode_rewards)
            #     eval_reward = evaluate_agent(env, agent, state_filter, n_starts=n_evals)
            #     writer.add_scalar('Reward/Train', running_reward, cumulative_timestep)
            #     writer.add_scalar('Reward/Test', eval_reward, cumulative_timestep)
            #     print('Episode {} \t Samples {} \t Avg length: {} \t Test reward: {} \t Train reward: {} \t Number of Updates: {}'.format(i_episode, samples_number, avg_length, eval_reward, running_reward, n_updates))
            #     episode_steps = []
            #     episode_rewards = []
            # if cumulative_timestep % gif_interval == 0:
            #     make_gif(agent, env, cumulative_timestep, state_filter)
            #     if save_model:
            #         make_checkpoint(agent, cumulative_timestep, params['env'])


        episode_steps.append(time_step)
        episode_rewards.append(episode_reward)


def evaluate_agent(env, agent, state_filter, n_starts=1):
    reward_sum = 0
    for _ in range(n_starts):
        done = False
        state = env.reset()
        while (not done):
            action = agent.get_action(state, state_filter=state_filter, deterministic=True)
            nextstate, reward, done, _ = env.step(action)
            reward_sum += reward
            state = nextstate
    return reward_sum / n_starts



#--------------------------------------------------------------------------------------------
# 以下函数为独立运行TD3时使用
#--------------------------------------------------------------------------------------------


def main():
    
    parser = ArgumentParser()
    parser.add_argument('--env', type=str, default='Humanoid-v2')
    parser.add_argument('--seed', type=int, default=100)
    parser.add_argument('--use_obs_filter', dest='obs_filter', action='store_true')
    parser.add_argument('--update_every_n_steps', type=int, default=1)
    parser.add_argument('--n_random_actions', type=int, default=25000)
    parser.add_argument('--n_collect_steps', type=int, default=1000)
    parser.add_argument('--n_evals', type=int, default=1)
    parser.add_argument('--save_model', dest='save_model', action='store_true')
    parser.set_defaults(obs_filter=False)
    parser.set_defaults(save_model=True)

    args = parser.parse_args()
    params = vars(args)

    seed = params['seed']
    env = gym.make(params['env'])
    env = RescaleAction(env, -1, 1)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = TD3_Agent(seed, state_dim, action_dim)

    train_agent_model_free(agent=agent, env=env, params=params)


if __name__ == '__main__':
    main()
