# python rlschedule_ppo.py --workload "./data/PIK-IPLEX-2009-1.swf" --exp_name pik1-ppo-t1024e100 --trajs 500 --seed 0 --epochs 100
# trajs-500 : cuda-out-of-memory -> trajs-200 try X 
# cpu로 진행 

import json
import joblib
import numpy as np
import argparse

import torch
from torch.optim import Adam
import torch.nn as nn
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

from sumtree_memory import SumTree


# gpu 사용 여부 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")


class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """
    e = 0.01
    a = 0.6
    beta = 0.4
    beta_increment_per_sampling = 0.001
    

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95): # lam = lambda
        self.size = size * 100  # assume the traj can be really long
        size = size * 100
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        # self.cobs_buf = np.zeros(combined_shape(size, JOB_SEQUENCE_SIZE*3), dtype=np.float32)
        self.cobs_buf = None
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.mask_buf = np.zeros(combined_shape(size, MAX_QUEUE_SIZE), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size
        self.tree = SumTree(size)
        # self.capacity = capacity
        self.max_adv = True # 추가 
        
    # PER 위해 추가
    def _get_priority(self, error):
        return (np.abs(error) + self.e) ** self.a

    # def __len__(self):
    #     if self.full:
    #         return self.buffer_size
    #     else:
    #         return self.ptr

    def store(self, obs, cobs, act, mask, rew, val, logp): # 이건 그냥 놔둠
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size  # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        # self.cobs_buf[self.ptr] = cobs
        self.act_buf[self.ptr] = act
        self.mask_buf[self.ptr] = mask
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0): # 한 epocch 끝나면 모두 계산
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr) 
        rews = np.append(self.rew_buf[path_slice], last_val) 
        vals = np.append(self.val_buf[path_slice], last_val)

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)

        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]
        
        # self.add() # ! 추가

        self.path_start_idx = self.ptr
    
    # 다시 생각해봐야함 add가 2개임
    def add(self): 
        for i in range(self.ptr):
            traj_p = np.max(np.abs(self.adv_buf[i])) if self.max_adv else np.mean(self.adv_buf[i])
            traj_p = self._get_priority(traj_p)
            self.tree.add(traj_p, (self.obs_buf[i], self.act_buf[i], self.mask_buf[i], self.ret_buf[i], self.adv_buf[i], self.logp_buf[i]))

    # random으로 뽑는 get 메서드 
    # 다 가져오는 듯? 그럼 좀 더 많이 뽑은 뒤 선별하는 방법으로 고려 
    # get -> get_sample
    def get_sample(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr < self.max_size
        actual_size = self.ptr
        self.ptr, self.path_start_idx = 0, 0

        actual_adv_buf = np.array(self.adv_buf, dtype=np.float32)
        actual_adv_buf = actual_adv_buf[:actual_size]

        adv_mean, adv_std = mpi_statistics_scalar(actual_adv_buf)
        """
        # This code is doing the advantage normalization trick; should be 
        # print ("-----------------------> actual_adv_buf: ", actual_adv_buf)
        adv_sum = np.sum(actual_adv_buf)
        adv_n = len(actual_adv_buf)
        adv_mean = adv_sum / adv_n
        adv_sum_sq = np.sum((actual_adv_buf - adv_mean) ** 2)
        adv_std = np.sqrt(adv_sum_sq / adv_n)
        # print ("-----------------------> adv_std:", adv_std)
        """
        actual_adv_buf = (actual_adv_buf - adv_mean) / adv_std

        data = dict(
            obs=self.obs_buf[:actual_size],
            act=self.act_buf[:actual_size],
            mask=self.mask_buf[:actual_size],
            ret=self.ret_buf[:actual_size],
            adv=actual_adv_buf,
            logp=self.logp_buf[:actual_size],
        )

        return {k: torch.as_tensor(v, dtype=torch.float32, device=device) for k, v in data.items()}

        # return [self.obs_buf[:actual_size], self.act_buf[:actual_size], self.mask_buf[:actual_size], actual_adv_buf, self.ret_buf[:actual_size], self.logp_buf[:actual_size]]
        
    def sample(self):
        assert self.ptr < self.max_size
        actual_size = self.ptr
        self.ptr, self.path_start_idx = 0, 0
        
        actual_adv_buf_ori = np.array(self.adv_buf, dtype=np.float32)
        actual_adv_buf_ori = actual_adv_buf_ori[:actual_size]

        adv_mean_ori, adv_std_ori = mpi_statistics_scalar(actual_adv_buf_ori)
        # normalize_advantage 
        actual_adv_buf_ori = (actual_adv_buf_ori - adv_mean_ori) / adv_std_ori
        
        data_ori = dict(
            obs=torch.as_tensor(self.obs_buf[:actual_size], dtype=torch.float32, device=device),
            act=torch.as_tensor(self.act_buf[:actual_size], dtype=torch.float32, device=device),
            mask=torch.as_tensor(self.mask_buf[:actual_size], dtype=torch.float32, device=device),
            ret=torch.as_tensor(self.ret_buf[:actual_size], dtype=torch.float32, device=device),
            adv=torch.as_tensor(actual_adv_buf_ori, dtype=torch.float32, device=device) ,
            logp=torch.as_tensor(self.logp_buf[:actual_size], dtype=torch.float32, device=device),
        )
        
        
        batch = []
        idxs = []
        segment = self.tree.total() / actual_size
        priorities = []

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(actual_size):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)
        
        sample_obs_buf = np.zeros(combined_shape(self.size, self.obs_dim), dtype=np.float32)
        # self.cobs_buf = np.zeros(combined_shape(size, JOB_SEQUENCE_SIZE*3), dtype=np.float32)
        # sample_cobs_buf = None
        sample_act_buf = np.zeros(combined_shape(self.size, self.act_dim), dtype=np.float32)
        sample_mask_buf = np.zeros(combined_shape(self.size, MAX_QUEUE_SIZE), dtype=np.float32)
        # sample_rew_buf = np.zeros(self.size, dtype=np.float32)
        sample_ret_buf = np.zeros(self.size, dtype=np.float32)
        sample_adv_buf = np.zeros(self.size, dtype=np.float32)
        # sample_val_buf = np.zeros(self.size, dtype=np.float32)
        sample_logp_buf = np.zeros(self.size, dtype=np.float32)
        
        for i in range(actual_size):
            
            # single trajectory 
            (traj_obs, traj_act, traj_mask, traj_ret, traj_adv, traj_logp) = batch[i]
            
            sample_obs_buf[i] = np.array(traj_obs).copy()
            # self.cobs_buf[i] = np.array(traj_cobs).copy()
            sample_act_buf[i] = np.array(traj_act).copy()
            sample_mask_buf[i] = np.array(traj_mask).copy()
            # sample_rew_buf[i] = np.array(traj_rew).copy()
            sample_ret_buf[i] = np.array(traj_ret).copy()
            sample_adv_buf[i] = np.array(traj_adv).copy()
            # sample_val_buf[i] = np.array(traj_val).copy()
            sample_logp_buf[i] = np.array(traj_logp).copy()
        
        actual_adv_buf = np.array(sample_adv_buf, dtype=np.float32)
        # actual_adv_buf = actual_adv_buf[:actual_size] # 수정해야함

        adv_mean, adv_std = mpi_statistics_scalar(actual_adv_buf)
        """
        # This code is doing the advantage normalization trick; should be 
        # print ("-----------------------> actual_adv_buf: ", actual_adv_buf)
        adv_sum = np.sum(actual_adv_buf)
        adv_n = len(actual_adv_buf)
        adv_mean = adv_sum / adv_n
        adv_sum_sq = np.sum((actual_adv_buf - adv_mean) ** 2)
        adv_std = np.sqrt(adv_sum_sq / adv_n)
        # print ("-----------------------> adv_std:", adv_std)
        """
        actual_adv_buf = (actual_adv_buf - adv_mean) / adv_std
        
        sample_obs_buf = torch.as_tensor(sample_obs_buf, dtype=torch.float32, device=device)
        sample_act_buf = torch.as_tensor(sample_act_buf, dtype=torch.float32, device=device)
        sample_mask_buf = torch.as_tensor(sample_mask_buf, dtype=torch.float32, device=device)
        sample_ret_buf = torch.as_tensor(sample_ret_buf, dtype=torch.float32, device=device)
        sample_adv_buf = torch.as_tensor(actual_adv_buf, dtype=torch.float32, device=device)
        sample_logp_buf = torch.as_tensor(sample_logp_buf, dtype=torch.float32, device=device)
        
        data = dict(
            obs=sample_obs_buf,
            act=sample_act_buf,
            mask=sample_mask_buf,
            ret=sample_ret_buf,
            adv=sample_adv_buf,
            logp=sample_logp_buf,
        )
        # 기본 참고
        # data = dict(
        #     obs=self.obs_buf[:actual_size],
        #     act=self.act_buf[:actual_size],
        #     mask=self.mask_buf[:actual_size],
        #     ret=self.ret_buf[:actual_size],
        #     adv=actual_adv_buf,
        #     logp=self.logp_buf[:actual_size],
        # )

        data.update(data_ori) #
        
        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()
        

        return data, idxs, is_weight


    def update(self, idx, error):
        error = error.cpu().numpy()
        error = np.max(np.abs(error)) if self.max_adv else np.mean(np.abs(error))
        p = self._get_priority(error)
        self.tree.update(idx, p)


"""
Network configurations
"""


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


class RLActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        # hidden_sizes = (32, 16)
        # self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)
        self.dense1 = nn.Linear(JOB_FEATURES, 32)
        self.dense2 = nn.Linear(32, 16)
        self.lstm1 = nn.LSTM(16, 8, batch_first=True)  # 첫 번째 LSTM 레이어 추가
        self.dense3 = nn.Linear(8, 8)
        self.dense4 = nn.Linear(16 + 8, 1)  # LSTM 결과를 Concatenate 하기 위해 차원 조정

    def _distribution(self, obs, mask):
        mask = torch.tensor(mask, device=device) # 추가
        x = obs.view(-1, MAX_QUEUE_SIZE, JOB_FEATURES)
        x = torch.relu(self.dense1(x))
        x = torch.relu(self.dense2(x))
        x_lstm, _ = self.lstm1(x)  # 첫 번째 LSTM 레이어 사용
        x_lstm = x_lstm[:, -1, :]  # 시퀀스의 마지막 출력 선택
        x = torch.relu(self.dense3(x))
        logits = torch.squeeze(self.dense4(x), -1)
        # logits = self.logits_net(obs)
        logits = logits + (mask - 1) * 1000000
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)

    def forward(self, obs, mask, act=None):
        # Produce action distributions for given observations, and
        # optionally compute the log likelihood of given actions under
        # those distributions.
        # print("obs shape: ", obs.shape)
        # print("mask shape: ", mask.shape)
        x = obs.view(-1, MAX_QUEUE_SIZE, JOB_FEATURES)
        x = torch.relu(self.dense1(x))
        x_dense = torch.relu(self.dense2(x))
        
        x_lstm, _ = self.lstm1(x_dense)  # 첫 번째 LSTM 레이어 사용
        x_lstm = x_lstm[:, -1, :]  # 시퀀스의 마지막 출력 선택
        x_lstm = torch.relu(self.dense3(x_lstm))
          
        x = torch.cat((x_dense[:, -1, :], x_lstm), dim=-1)
        logits = self.dense4(x).squeeze(-1)
        mask = (mask - 1) * 1000000
        logits = logits + mask
        pi = Categorical(logits=logits)

        logp_a = None
        if act is not None:
            logp_a = pi.log_prob(act)
        return pi, logp_a


class RLCritic(nn.Module):
    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        hidden_sizes = (32, 16, 8)
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, mask):
        return torch.squeeze(
            self.v_net(obs), -1
        )  # Critical to ensure v has right shape.


class RLActorCritic(nn.Module):
    def __init__(
        self, observation_space, action_space, hidden_sizes=(64, 64), activation=nn.ReLU
    ):
        super().__init__()
        obs_dim = observation_space.shape[0]
        # build actor function
        self.pi = RLActor(obs_dim, action_space.n, hidden_sizes, activation)
        # build value function
        self.v = RLCritic(obs_dim, hidden_sizes, activation)

    def step(self, obs, mask):
        with torch.no_grad():
            pi = self.pi._distribution(obs, mask)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs, mask)
        return a.cpu().numpy(), v.cpu().numpy(), logp_a.cpu().numpy() # cpu 추가

    # def act(self, obs):
    #     return self.step(obs)[0] 
    def act(self, obs, mask):
        return self.step(obs, mask)[0] # 수정


"""
Proximal Policy Optimization (by clipping), 

with early stopping based on approximate KL

"""


def ppo(
    workload_file,
    model_path,
    ac_kwargs=dict(),
    seed=0,
    traj_per_epoch=4000,
    epochs=50,
    gamma=0.99,
    clip_ratio=0.2,
    pi_lr=3e-4,
    vf_lr=1e-3,
    train_pi_iters=80,
    train_v_iters=80,
    lam=0.97,
    max_ep_len=1000,
    target_kl=0.01,
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
    # Special function to avoid certain slowdowns from PyTorch + MPI combo.
    setup_pytorch_for_mpi()

    # Set up logger and save configuration
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    # Random seed
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

    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape

    # Create actor-critic module
    ac = RLActorCritic(env.observation_space, env.action_space, **ac_kwargs).to(device) # gpu 추가

    # Sync params across processes
    sync_params(ac)

    # Share information about action space with policy architecture
    ac_kwargs["action_space"] = env.action_space
    ac_kwargs["attn"] = attn

    # Count variables
    var_counts = tuple(count_vars(module) for module in [ac.pi, ac.v])
    logger.log("\nNumber of parameters: \t pi: %d, \t v: %d\n" % var_counts)

    # Inputs to computation graph

    local_traj_per_epoch = int(traj_per_epoch / num_procs())
    buf = PPOBuffer(
        obs_dim, act_dim, local_traj_per_epoch * JOB_SEQUENCE_SIZE, gamma, lam
    )

    # Set up function for computing PPO policy loss
    def compute_loss_pi(data):
        obs, act, adv, logp_old, mask = (
            data["obs"],
            data["act"],
            data["adv"],
            data["logp"],
            data["mask"],
        )

        # Policy loss
        pi, logp = ac.pi(obs, mask, act) 
        ratio = torch.exp(logp - logp_old).to(device)
        clip_adv = (torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv)# .cpu().numpy() # .to(device)
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean() 

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item() 
        clipped = ratio.gt(1 + clip_ratio) | ratio.lt(1 - clip_ratio) 
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item() 
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac) 

        return loss_pi, pi_info

    # Set up function for computing value loss
    def compute_loss_v(data):
        obs, ret, mask = data["obs"], data["ret"], data["mask"]
        return ((ac.v(obs, mask) - ret) ** 2).mean().to(device)

    # Set up optimizers for policy and value function
    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    vf_optimizer = Adam(ac.v.parameters(), lr=vf_lr)

    # Set up model saving
    logger.setup_pytorch_saver(ac)

    def update_learn():
        data, idxs, is_weight = buf.sample() # 배치 형태로 가져와야 함

        pi_l_old, pi_info_old = compute_loss_pi(data)
        pi_l_old = pi_l_old.item()
        v_l_old = compute_loss_v(data).item()

        # Train policy with multiple steps of gradient descent
        for i in range(train_pi_iters):
            pi_optimizer.zero_grad()
            loss_pi, pi_info = compute_loss_pi(data)
            kl = mpi_avg(pi_info["kl"])
            if kl > 1.5 * target_kl:
                logger.log("Early stopping at step %d due to reaching max kl." % i)
                break
            loss_pi.backward()
            mpi_avg_grads(ac.pi)  # average grads across MPI processes
            pi_optimizer.step()

        logger.store(StopIter=i)

        # Value function learning
        for i in range(train_v_iters):
            vf_optimizer.zero_grad()
            loss_v = compute_loss_v(data)
            loss_v.backward()
            mpi_avg_grads(ac.v)  # average grads across MPI processes
            vf_optimizer.step()

        # Log changes from update
        kl, ent, cf = pi_info["kl"], pi_info_old["ent"], pi_info["cf"]
        # 추가해야함
        imp_sample_ratio = kl
        # print(imp_sample_ratio)
        
        
        logger.store(
            LossPi=pi_l_old,
            LossV=v_l_old,
            KL=kl,
            Entropy=ent,
            ClipFrac=cf,
            DeltaLossPi=(loss_pi.item() - pi_l_old),
            DeltaLossV=(loss_v.item() - v_l_old),
        )
        
        for i in range(len(data)): # 여기 확인해봐야함
            idx = idxs[i]
            buf.update(idx, data['adv'][i])
        
        # update_lean() 함수 끝

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

    # Main loop: collect experience in env and update/log each epoch
    start_time = MPI.Wtime()
    num_total = 0
    for epoch in range(epochs):
        t = 0
        while True:
            lst = []
            for i in range(0, MAX_QUEUE_SIZE * JOB_FEATURES, JOB_FEATURES):
                if all(o[i : i + JOB_FEATURES] == [0] + [1] * (JOB_FEATURES - 2) + [0]):
                    lst.append(0)
                elif all(o[i : i + JOB_FEATURES] == [1] * JOB_FEATURES):
                    lst.append(0)
                else:
                    lst.append(1)

            a, v_t, logp_t = ac.step(
                torch.as_tensor(o, dtype=torch.float32, device=device), np.array(lst).reshape(1, -1) 
            ) # gpu 추가
            # a, v_t, logp_t = ac.step(torch.as_tensor(o, dtype=torch.float32), np.array(lst).reshape(1,-1))

            num_total += 1
            """
            action = np.random.choice(np.arange(MAX_QUEUE_SIZE), p=action_probs)
            log_action_prob = np.log(action_probs[action])
            """

            # save and log
            buf.store(o, None, a, np.array(lst), r, v_t, logp_t)
            logger.store(VVals=v_t)

            o, r, d, r2, sjf_t, f1_t = env.step(a[0])
            ep_ret += r
            ep_len += 1
            show_ret += r2
            sjf += sjf_t
            f1 += f1_t

            if d: # true이면 즉, 끝나면 
                t += 1
                buf.finish_path(r)
                
                logger.store(
                    EpRet=ep_ret, EpLen=ep_len, ShowRet=show_ret, SJF=sjf, F1=f1
                )
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
                if t >= local_traj_per_epoch:
                    # print ("state:", state, "\nlast action in a traj: action_probs:\n", action_probs, "\naction:", action)
                    break
        # print("Sample time:", (time.time()-start_time)/num_total, num_total)
        # Save model
        if (epoch % save_freq == 0) or (epoch == epochs - 1):
            logger.save_state({"env": env}, None)

        buf.add()
        # start_time = time.time()
        update_learn()
        # print("Train time:", time.time()-start_time)

        # Log info about epoch
        logger.log_tabular("Epoch", epoch)
        logger.log_tabular("EpRet", with_min_and_max=True)
        logger.log_tabular("EpLen", with_min_and_max=True)
        logger.log_tabular("VVals", with_min_and_max=True)
        logger.log_tabular(
            "TotalEnvInteracts", (epoch + 1) * traj_per_epoch * JOB_SEQUENCE_SIZE
        )
        logger.log_tabular("LossPi", average_only=True)
        logger.log_tabular("LossV", average_only=True)
        logger.log_tabular("DeltaLossPi", average_only=True)
        logger.log_tabular("DeltaLossV", average_only=True)
        logger.log_tabular("Entropy", average_only=True)
        logger.log_tabular("KL", average_only=True)
        logger.log_tabular("ClipFrac", average_only=True)
        logger.log_tabular("StopIter", average_only=True)
        logger.log_tabular("ShowRet", average_only=True)
        logger.log_tabular("SJF", average_only=True)
        logger.log_tabular("F1", average_only=True)
        logger.log_tabular("Time", MPI.Wtime() - start_time)
        logger.dump_tabular()


if __name__ == "__main__":
    # test_job_workload();
    # test_hpc_env()

    """
    actual training code
    """
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

    mpi_fork(args.cpu)  # run parallel code with mpi

    # build absolute path for using in hpc_env.
    current_dir = os.getcwd()
    workload_file = os.path.join(current_dir, args.workload)
    log_data_dir = os.path.join(current_dir, "./logs/")
    logger_kwargs = setup_logger_kwargs(
        args.exp_name, seed=args.seed, data_dir=log_data_dir
    )

    ppo(
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
