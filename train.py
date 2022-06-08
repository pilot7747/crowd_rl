# from imitation.algorithms import preference_comparisons
import preference_comparisons
from reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm
from imitation.policies.base import FeedForward32Policy, NormalizeFeaturesExtractor
import seals
import gym
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO
import argparse
import warnings
import mujoco_py
warnings.filterwarnings("ignore")

import abc
import math
import pickle
import random
from typing import Any, Callable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch as th
from scipy import special
from stable_baselines3.common import base_class, vec_env

from imitation.algorithms import base
from imitation.data import rollout, types, wrappers
from imitation.data.types import (
    AnyPath,
    TrajectoryPair,
    TrajectoryWithRew,
    TrajectoryWithRewPair,
    Transitions,
)
from imitation.policies import exploration_wrapper
from imitation.rewards import common as rewards_common
from imitation.rewards import reward_wrapper
import reward_nets
from imitation.util import logger as imit_logger
from imitation.util import networks

import logging
import boto3
from botocore.exceptions import ClientError
import os
import shutil
from tqdm.auto import tqdm
import toloka.client as toloka
from crowdkit.aggregation import MajorityVote
import time
from multiprocessing import Pool
from gym.wrappers import Monitor

import numpy as np



class TolokaGatherer(preference_comparisons.PreferenceGatherer):
    """Gatherer that uses Toloka to recieve a human feedback for trajectory pairs
    """
    def __init__(
        self,
        venv,
        path,
        aws_access_key_id,
        aws_secret_access_key,
        endpoint_url,
        bucket,
        toloka_token,
        base_pool,
        base_url,
        custom_logger: Optional[imit_logger.HierarchicalLogger] = None,
    ):
        super().__init__(custom_logger=custom_logger)
        self.iteration = 0
        self.venv = venv
        self.path = path
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.endpoint_url = endpoint_url
        self.bucket = bucket
        self.toloka_client = toloka.TolokaClient(toloka_token, 'PRODUCTION')
        self.base_pool = base_pool
        self.base_url = base_url
        
    def upload_file(self, file_name, object_name=None):
        """Uploads a file to S3
        """
        if object_name is None:
            object_name = os.path.basename(file_name)

        session = boto3.session.Session()

        s3_client = session.client(
            service_name='s3',
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
            endpoint_url=self.endpoint_url,
        )
        try:
            response = s3_client.upload_file(file_name, self.bucket, object_name)
        except ClientError as e:
            logging.error(e)
            return False
        return True
    
    def record_video(self, trajectory, path, output_filename):
        """Record a video of single trajectory
        """
        tmp_path = os.path.join(path, 'tmp_video')
        env = Monitor(gym.make(self.venv), tmp_path, force=True)
        _ = env.reset()
        initial_obs = trajectory.obs[0]
        initial_state = mujoco_py.MjSimState(time=0.0, qpos=initial_obs[:6], qvel=initial_obs[6:], act=None, udd_state={})
        env.unwrapped.sim.set_state(initial_state)
        for act in trajectory.acts:
            next_state, reward, done, _ = env.step(act)
        env.close()

        for file in os.listdir(tmp_path):
            if file.endswith('.mp4'):
                tmp_file = os.path.join(tmp_path, file)
                break

        shutil.move(tmp_file, output_filename)
        shutil.rmtree(tmp_path)


    def record_trajectory_pair(self, trajectory_1, trajectory_2, index):
        """Record videos of pair's trajectories
        """
        pair_path = os.path.join(self.path, str(index))
        os.mkdir(pair_path)
        self.record_video(trajectory_1, pair_path, os.path.join(pair_path, '0.mp4'))
        self.record_video(trajectory_2, pair_path, os.path.join(pair_path, '1.mp4'))


    def upload_files(self, iteration, n_comparisons):
        """Upload pair's videos to S3
        """
        for i in range(n_comparisons):
            self.upload_file(os.path.join(self.path, str(i), '0.mp4'), f'{iteration}_{i}_0.mp4')
            self.upload_file(os.path.join(self.path, str(i), '1.mp4'), f'{iteration}_{i}_1.mp4')

    def make_videos(self, iteration, comparisons):
        """Record trajectories videos and upload them to S3
        """
        os.mkdir(self.path)
        progress = tqdm(enumerate(comparisons), total=len(comparisons))
        progress.set_description('Recording clips')
        for i, pair in progress:
            self.record_trajectory_pair(*pair, i)
        self.upload_files(iteration, len(comparisons))
        shutil.rmtree(self.path)

    def wait_pool_for_close(self, pool_id, minutes_to_wait=1):
        """Utility function that waits until the pool is closed
        """
        sleep_time = 60 * minutes_to_wait
        pool = self.toloka_client.get_pool(pool_id)
        while not pool.is_closed():
            op = self.toloka_client.get_analytics([toloka.analytics_request.CompletionPercentagePoolAnalytics(subject_id=pool.id)])
            op = self.toloka_client.wait_operation(op)
            percentage = op.details['value'][0]['result']['value']
            print(f'Pool {pool.id} - {percentage}%')
            time.sleep(sleep_time)
            pool = self.toloka_client.get_pool(pool.id)

    def run_toloka_annotation(self, n_comparisons, iteration):
        """Clones pool, uploads the tasks and fetches annotation results
        """
        pool = self.toloka_client.clone_pool(pool_id=self.base_pool)
        pool.set_mixer_config(
            real_tasks_count=5,
            golden_tasks_count=0
        )
        pool.private_name = f'Iteration {iteration}'
        pool = self.toloka_client.update_pool(pool.id, pool)
        tasks = [
            toloka.Task(
                pool_id=pool.id,
                input_values={'video1': f'{self.base_url}/{iteration}_{i}_0.mp4', 'video2': f'{self.base_url}/{iteration}_{i}_1.mp4'},
            )
            for i in range(n_comparisons)
        ]
        created_tasks = self.toloka_client.create_tasks(tasks, allow_defaults=True)
        print('Tasks created')
        pool = self.toloka_client.open_pool(pool.id)
        pool_id = pool.id

        self.wait_pool_for_close(pool_id)


        answers_df = self.toloka_client.get_assignments_df(pool_id)

        answers_df['task'] = answers_df.apply(lambda row: row['INPUT:video1'].split('/')[-1] + '\t' + row['INPUT:video2'].split('/')[-1], axis=1)
        agg_df = answers_df[['task', 'ASSIGNMENT:worker_id', 'OUTPUT:result']]
        agg_df.columns = ['task', 'worker', 'label']
        agg_res = MajorityVote().fit_predict(agg_df)

        result = []
        for i in range(n_comparisons):
            task = f'{iteration}_{i}_0.mp4\t{iteration}_{i}_1.mp4'
            label = agg_res[task]
            if label == 'left':
                result.append(1)
            else:
                result.append(0)

        return np.array(result).astype(np.float32)
    
    def __call__(self, fragment_pairs: Sequence[TrajectoryWithRewPair]) -> np.ndarray:
        """Computes probability fragment 1 is preferred over fragment 2."""
        self.make_videos(self.iteration, fragment_pairs)
        result = self.run_toloka_annotation(len(fragment_pairs), self.iteration)
        self.iteration += 1
        return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gatherer', type=str, default='synth')
    parser.add_argument('--toloka-token', type=str)
    parser.add_argument('--aws-key-id', type=str)
    parser.add_argument('--connection-url', type=str)
    parser.add_argument('--aws-secret-access', type=str)
    parser.add_argument('--bucket', type=str)
    parser.add_argument('--base-pool-id', type=int)
    parser.add_argument('--base-url', type=str)
    cmd_args = parser.parse_args()

    venv = DummyVecEnv([lambda: gym.make("seals/Hopper-v0")] * 8)

    reward_net = BasicRewardNet(
        venv.observation_space, venv.action_space, normalize_input_layer=RunningNorm
    )

    fragmenter = preference_comparisons.RandomFragmenter(warning_threshold=0, seed=0)

    if cmd_args.gatherer != 'synth':
        
        gatherer = TolokaGatherer(
            "seals/Hopper-v0",
            'pairs',
            cmd_args.aws_key_id,
            cmd_args.aws_secret_access,
            cmd_args.connection_url,
            cmd_args.bucket,
            cmd_args.toloka_token,
            cmd_args.base_pool_id,
            cmd_args.base_url
        )
    else:
        gatherer = preference_comparisons.SyntheticGatherer(seed=0)

    reward_trainer = preference_comparisons.CrossEntropyRewardTrainer(
        model=reward_net,
        epochs=3,
    )

    agent = PPO(
        policy=FeedForward32Policy,
        policy_kwargs=dict(
            features_extractor_class=NormalizeFeaturesExtractor,
            features_extractor_kwargs=dict(normalize_class=RunningNorm),
        ),
        env=venv,
        seed=0,
        n_steps=2048 // venv.num_envs,
        batch_size=1024,
        ent_coef=0.0,
        learning_rate=0.0003,
        n_epochs=10,
        tensorboard_log="./runs/",
    )

    agent.set_env(venv)

    trajectory_generator = preference_comparisons.AgentTrainer(
        algorithm=agent,
        reward_fn=reward_net,
        exploration_frac=0.0,
        seed=0,
    )

    pref_comparisons = preference_comparisons.PreferenceComparisons(
        trajectory_generator,
        reward_net,
        fragmenter=fragmenter,
        preference_gatherer=gatherer,
        reward_trainer=reward_trainer,
        comparisons_per_iteration=100,
        fragment_length=150,
        transition_oversampling=1,
        initial_comparison_frac=0.1,
        allow_variable_horizon=False,
        seed=0,
        initial_epoch_multiplier=20,
    )

    pref_comparisons.train(
        total_timesteps=12000000,
        total_comparisons=9000,
    )

    th.save(reward_net.state_dict(), 'reward_net_hop.pt')
    agent.save('ppo_agent_hop')
