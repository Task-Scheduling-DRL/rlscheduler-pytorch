import json
import joblib
import numpy as np
import argparse

import torch
from torch.optim import Adam
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

import scipy.signal

import os.path as osp, time, atexit, os
import warnings
from copy import deepcopy
import os, subprocess, sys
import string
from subprocess import CalledProcessError
from textwrap import dedent
import time

from mpi4py import MPI
from schedgym import *

from util_pytorch import *


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for DDPG agents.
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        size = size * 100  # assume the traj can be really long
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        # self.cobs_buf = np.zeros(combined_shape(size, JOB_SEQUENCE_SIZE*3), dtype=np.float32)
        self.cobs_buf = None
        self.obs2_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, cobs, act, rew, next_obs, done):
        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        # self.cobs_buf[self.ptr] = cobs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(
            obs=self.obs_buf[idxs],
            obs2=self.obs2_buf[idxs],
            act=self.act_buf[idxs],
            rew=self.rew_buf[idxs],
            done=self.done_buf[idxs],
        )
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in batch.items()}


class RLActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super(RLActor, self).__init__()
        # pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
        # self.pi = mlp(pi_sizes, activation, nn.Tanh)

        self.dense1 = nn.Linear(JOB_FEATURES, 32)
        self.dense2 = nn.Linear(32, 16)
        self.dense3 = nn.Linear(16, 8)
        self.dense4 = nn.Linear(8, 1)

    def forward(self, obs):
        # Return output from network scaled to action space limits.
        x = obs.view(-1, MAX_QUEUE_SIZE, JOB_FEATURES)
        x = torch.relu(self.dense1(x))
        x = torch.relu(self.dense2(x))
        x = torch.relu(self.dense3(x))
        x = torch.relu(x)
        gumbel_dist = torch.distributions.gumbel.Gumbel(
            0.0, 1.0
        )  # create a gumbel distribution
        gumbel_noise = gumbel_dist.sample(x.shape)  # sample gumbel noise
        x = x + gumbel_noise  # add gumbel noise to logits
        x = F.softmax(x / 0.01, dim=-1)  # apply softmax with temperature 0.01
        x = torch.argmax(x, dim=-1)  # apply argmax to get discrete actions
        return x


class RLQFunction(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super(RLQFunction, self).__init__()
        hidden_sizes = (32, 16, 8)
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1)  # Critical to ensure q has right shape.


class RLActorCritic(nn.Module):
    def __init__(
        self, observation_space, action_space, hidden_sizes=(64, 64), activation=nn.ReLU
    ):
        super().__init__()
        obs_dim = observation_space.shape[0]
        # print(observation_space.shape)
        # print(obs_dim)
        act_dim = action_space.shape  # [0]
        # print(action_space)
        # print(action_space.shape)
        # print(action_space.n)

        # build policy and value functions
        self.pi = RLActor(obs_dim, action_space.n, hidden_sizes, activation)
        self.q = RLQFunction(obs_dim, action_space.n, hidden_sizes, activation)

    def act(self, obs):
        with torch.no_grad():
            return self.pi(obs).numpy()


# class RLActor(nn.Module):
#     def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
#         super(RLActor, self).__init__()
#         # pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
#         # self.pi = mlp(pi_sizes, activation, nn.Tanh)
#         self.act_limit = act_limit

#         self.dense1 = nn.Linear(JOB_FEATURES, 32)
#         self.dense2 = nn.Linear(32, 16)
#         self.dense3 = nn.Linear(16, 8)
#         self.dense4 = nn.Linear(8, 1)

#     def forward(self, obs):
#         # Return output from network scaled to action space limits.
#         x = obs.view(-1, MAX_QUEUE_SIZE, JOB_FEATURES)
#         x = torch.relu(self.dense1(x))
#         x = torch.relu(self.dense2(x))
#         x = torch.relu(self.dense3(x))
#         x = torch.relu(self.dense4(x)) * self.act_limit # tanh -> relu
#         return x

# class RLQFunction(nn.Module):
#     def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
#         super(RLQFunction, self).__init__()
#         hidden_sizes = (32, 16, 8)
#         self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

#     def forward(self, obs, act):
#         q = self.q(torch.cat([obs, act], dim=-1))
#         return torch.squeeze(q, -1) # Critical to ensure q has right shape.

# class RLActorCritic(nn.Module):

#     def __init__(self, observation_space, action_space,
#                  hidden_sizes=(64,64),
#                  activation=nn.ReLU):
#         super().__init__()
#         obs_dim = observation_space.shape[0]
#         print(observation_space.shape)
#         print(obs_dim)
#         act_dim = action_space.shape # [0]
#         print(action_space)
#         print(action_space.shape)
#         print(action_space.n)
#         act_limit = action_space.high[0] # 확인해야함

#         # build policy and value functions
#         self.pi = RLActor(obs_dim, act_dim, hidden_sizes, activation, act_limit)
#         self.q = RLQFunction(obs_dim, act_dim, hidden_sizes, activation)

#     def act(self, obs):
#         with torch.no_grad():
#             return self.pi(obs).numpy()


def ddpg(
    workload_file,
    model_path,
    ac_kwargs=dict(),
    seed=0,
    traj_per_epoch=4000,
    epochs=50,
    replay_size=int(1e4),
    gamma=0.99,
    polyak=0.995,
    pi_lr=1e-3,
    q_lr=1e-3,
    batch_size=100,
    start_steps=10000,
    update_after=1000,
    update_every=50,
    act_noise=0.1,
    num_test_episodes=10,
    max_ep_len=1000,
    logger_kwargs=dict(),
    save_freq=10,
    pre_trained=0,
    trained_model=None,
    attn=False,
    shuffle=False,
    backfil=False,
    skip=False,
    score_type=0,
    batch_job_slice=0,
):
    """
    Deep Deterministic Policy Gradient (DDPG)


    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: The constructor method for a PyTorch Module with an ``act``
            method, a ``pi`` module, and a ``q`` module. The ``act`` method and
            ``pi`` module should accept batches of observations as inputs,
            and ``q`` should accept a batch of observations and a batch of
            actions as inputs. When called, these should return:

            ===========  ================  ======================================
            Call         Output Shape      Description
            ===========  ================  ======================================
            ``act``      (batch, act_dim)  | Numpy array of actions for each
                                           | observation.
            ``pi``       (batch, act_dim)  | Tensor containing actions from policy
                                           | given observations.
            ``q``        (batch,)          | Tensor containing the current estimate
                                           | of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object
            you provided to DDPG.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs)
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs to run and train agent.

        replay_size (int): Maximum length of replay buffer.

        gamma (float): Discount factor. (Always between 0 and 1.)

        polyak (float): Interpolation factor in polyak averaging for target
            networks. Target networks are updated towards main networks
            according to:

            .. math:: \\theta_{\\text{targ}} \\leftarrow
                \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta

            where :math:`\\rho` is polyak. (Always between 0 and 1, usually
            close to 1.)

        pi_lr (float): Learning rate for policy.

        q_lr (float): Learning rate for Q-networks.

        batch_size (int): Minibatch size for SGD.

        start_steps (int): Number of steps for uniform-random action selection,
            before running real policy. Helps exploration.

        update_after (int): Number of env interactions to collect before
            starting to do gradient descent updates. Ensures replay buffer
            is full enough for useful updates.

        update_every (int): Number of env interactions that should elapse
            between gradient descent updates. Note: Regardless of how long
            you wait between updates, the ratio of env steps to gradient steps
            is locked to 1.

        act_noise (float): Stddev for Gaussian exploration noise added to
            policy at training time. (At test time, no noise is added.)

        num_test_episodes (int): Number of episodes to test the deterministic
            policy at the end of each epoch.

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    """

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    seed += 10000 * proc_id()
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = SchedGym(
        shuffle=shuffle,
        backfil=backfil,
        skip=skip,
        job_score_type=score_type,
        batch_job_slice=batch_job_slice,
        build_sjf=False,
    )
    env.seed(seed)
    env.my_init(workload_file=workload_file, sched_file=model_path)
    test_env = SchedGym(
        shuffle=shuffle,
        backfil=backfil,
        skip=skip,
        job_score_type=score_type,
        batch_job_slice=batch_job_slice,
        build_sjf=False,
    )
    test_env.seed(seed)
    test_env.my_init(workload_file=workload_file, sched_file=model_path)

    obs_dim = env.observation_space.shape
    # act_dim = env.action_space.shape[0] # 확인해야함
    act_dim = env.action_space.shape
    # print('act_dim: ', act_dim)

    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    # act_limit = env.action_space.high[0] # 확인해야함
    # act_limit = env.action_space.shape # 확인해야함

    # Share information about action space with policy architecture
    # ac_kwargs['action_space'] = env.action_space
    # ac_kwargs['attn'] = attn

    # Create actor-critic module and target networks
    ac = RLActorCritic(env.observation_space, env.action_space, **ac_kwargs)
    ac_targ = deepcopy(ac)

    # Count variables
    var_counts = tuple(count_vars(module) for module in [ac.pi, ac.q])
    logger.log("\nNumber of parameters: \t pi: %d, \t v: %d\n" % var_counts)

    # Freeze target networks with respect to optimizers (only update via polyak averaging)
    for p in ac_targ.parameters():
        p.requires_grad = False

    # Experience buffer
    local_traj_per_epoch = int(traj_per_epoch / num_procs())
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    # Set up function for computing DDPG Q-loss
    def compute_loss_q(data):
        obs, act, rew, obs2, done = (
            data["obs"],
            data["act"],
            data["rew"],
            data["obs2"],
            data["done"],
        )

        q = ac.q(obs, act)

        # Bellman backup for Q function
        with torch.no_grad():
            q_pi_targ = ac_targ.q(obs2, ac_targ.pi(obs2))
            backup = rew + gamma * (1 - done) * q_pi_targ

        # MSE loss against Bellman backup
        loss_q = ((q - backup) ** 2).mean()

        # Useful info for logging
        loss_info = dict(QVals=q.detach().numpy())  # 확인해야함

        return loss_q, loss_info

    # Set up function for computing DDPG pi loss
    def compute_loss_pi(data):
        obs = data["obs"]
        q_pi = ac.q(obs, ac.pi(obs))
        return -q_pi.mean()

    # Set up optimizers for policy and q-function
    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    q_optimizer = Adam(ac.q.parameters(), lr=q_lr)

    # Set up model saving
    logger.setup_pytorch_saver(ac)

    def update(data):
        # First run one gradient descent step for Q.
        q_optimizer.zero_grad()
        loss_q, loss_info = compute_loss_q(data)
        loss_q.backward()
        q_optimizer.step()

        # Freeze Q-network so you don't waste computational effort
        # computing gradients for it during the policy learning step.
        for p in ac.q.parameters():
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        pi_optimizer.zero_grad()
        loss_pi = compute_loss_pi(data)
        loss_pi.backward()
        pi_optimizer.step()

        # Unfreeze Q-network so you can optimize it at next DDPG step.
        for p in ac.q.parameters():
            p.requires_grad = True

        # Record things
        logger.store(LossQ=loss_q.item(), LossPi=loss_pi.item(), **loss_info)

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)

    def get_action(o):
        a = ac.act(torch.as_tensor(o, dtype=torch.float32))
        return a 
        # a += noise_scale * np.random.randn(act_dim)
        # return np.clip(
        #     a, -act_dim, act_dim
        # )  # original: np.clip(a, -act_limit, act_limit)

    def test_agent():
        for j in range(num_test_episodes):
            o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
            while not (d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time (noise_scale=0)
                o, r, d, _ = test_env.step(get_action(o))
                ep_ret += r
                ep_len += 1
            logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

    # Prepare for interaction with environment
    total_steps = traj_per_epoch * epochs
    start_time = time.time()
    [o, co], r, d, ep_ret, ep_len, show_ret, sjf, f1 = (
        env.reset(),
        0,
        False,
        0,
        0,
        0,
        0,
        0,
    )

    # 수정해야함
    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):
        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards,
        # use the learned policy (with some noise, via act_noise).
        if t > start_steps:
            # a = get_action(o, act_noise)
            a = get_action(o)
        else:
            a = env.action_space.sample()

        # Step the env
        a = ac.act(a)

        # save and log
        buf.store(o, None, a, np.array(lst), r, v_t, logp_t)
        logger.store(VVals=v_t)

        o, r, d, r2, sjf_t, f1_t = env.step(a[0])
        ep_ret += r
        ep_len += 1
        show_ret += r2
        sjf += sjf_t
        f1 += f1_t

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len == max_ep_len else d

        # Store experience to replay buffer
        replay_buffer.store(o, None, a, r, o2, d)

        # Super critical, easy to overlook step: make sure to update
        # most recent observation!
        o = o2

        # End of trajectory handling
        if d or (ep_len == max_ep_len):
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            o, ep_ret, ep_len = env.reset(), 0, 0

        # Update handling
        if t >= update_after and t % update_every == 0:
            for _ in range(update_every):
                batch = replay_buffer.sample_batch(batch_size)
                update(data=batch)

        # End of epoch handling
        if (t + 1) % traj_per_epoch == 0:
            epoch = (t + 1) // traj_per_epoch

            # Save model
            if (epoch % save_freq == 0) or (epoch == epochs):
                logger.save_state({"env": env}, None)

            # Test the performance of the deterministic version of the agent.
            test_agent()

            # Log info about epoch
            logger.log_tabular("Epoch", epoch)
            logger.log_tabular("EpRet", with_min_and_max=True)
            logger.log_tabular("TestEpRet", with_min_and_max=True)
            logger.log_tabular("EpLen", average_only=True)
            logger.log_tabular("TestEpLen", average_only=True)
            logger.log_tabular("TotalEnvInteracts", t)
            logger.log_tabular("QVals", with_min_and_max=True)
            logger.log_tabular("LossPi", average_only=True)
            logger.log_tabular("LossQ", average_only=True)
            logger.log_tabular("Time", time.time() - start_time)
            logger.dump_tabular()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workload", type=str, default="./data/lublin_256.swf"
    )  # RICC-2010-2 lublin_256.swf SDSC-SP2-1998-4.2-cln.swf
    parser.add_argument("--model", type=str, default="./data/lublin_256.schd")
    parser.add_argument("--gamma", type=float, default=1)
    parser.add_argument("--seed", "-s", type=int, default=0)
    parser.add_argument("--cpu", type=int, default=1)
    parser.add_argument("--trajs", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--exp_name", type=str, default="ppo")
    parser.add_argument("--pre_trained", type=int, default=0)
    parser.add_argument(
        "--trained_model", type=str, default="./logs/ppo_temp/ppo_temp_s0"
    )
    parser.add_argument("--attn", type=int, default=0)
    parser.add_argument("--shuffle", type=int, default=0)
    parser.add_argument("--backfil", type=int, default=0)
    parser.add_argument("--skip", type=int, default=0)
    parser.add_argument("--score_type", type=int, default=0)
    parser.add_argument("--batch_job_slice", type=int, default=0)
    args = parser.parse_args()

    # build absolute path for using in hpc_env.
    current_dir = os.getcwd()
    workload_file = os.path.join(current_dir, args.workload)
    log_data_dir = os.path.join(current_dir, "./logs/")
    logger_kwargs = setup_logger_kwargs(
        args.exp_name, seed=args.seed, data_dir=log_data_dir
    )

    ddpg(
        workload_file,
        args.model,
        gamma=args.gamma,
        seed=args.seed,
        traj_per_epoch=args.trajs,
        epochs=args.epochs,
        logger_kwargs=logger_kwargs,
        pre_trained=0,
        attn=args.attn,
        shuffle=args.shuffle,
        backfil=args.backfil,
        skip=args.skip,
        score_type=args.score_type,
        batch_job_slice=args.batch_job_slice,
    )
