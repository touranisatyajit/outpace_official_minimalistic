import os
from random import random, uniform
os.environ['MUJOCO_GL'] = 'egl'
os.environ['EGL_DEVICE_ID'] = '0'


import copy
import pickle as pkl
import sys
import time

import numpy as np
from queue import Queue
import hydra
import torch
import utils
from logger import Logger
from replay_buffer import ReplayBuffer, HindsightExperienceReplayWrapperVer2
from video import VideoRecorder
import matplotlib.pyplot as plt
import seaborn as sns
from hgg.hgg import goal_distance
from visualize.visualize_2d import *
torch.backends.cudnn.benchmark = True

class UniformFeasibleGoalSampler:
    def __init__(self, env_name):        
        self.env_name = env_name        
        if env_name in ['AntMazeSmall-v0', 'PointUMaze-v0']:
            self.LOWER_CONTEXT_BOUNDS = np.array([-2, -2]) 
            self.UPPER_CONTEXT_BOUNDS = np.array([10, 10])
        elif env_name in ['sawyer_peg_pick_and_place']:
            self.LOWER_CONTEXT_BOUNDS = np.array([-0.6, 0.2, 0.01478]) 
            self.UPPER_CONTEXT_BOUNDS = np.array([0.6, 1.0, 0.4])            
        elif env_name in ['sawyer_peg_push']:
            self.LOWER_CONTEXT_BOUNDS = np.array([-0.6, 0.2, 0.01478]) 
            self.UPPER_CONTEXT_BOUNDS = np.array([0.6, 1.0, 0.02])        
        elif env_name == "PointSpiralMaze-v0":
            self.LOWER_CONTEXT_BOUNDS = np.array([-10, -10]) 
            self.UPPER_CONTEXT_BOUNDS = np.array([10, 10])
        elif env_name in ["PointNMaze-v0"]:
            self.LOWER_CONTEXT_BOUNDS = np.array([-2, -2]) 
            self.UPPER_CONTEXT_BOUNDS = np.array([10, 18])
        else:
            raise NotImplementedError

    def is_feasible(self, context): 
        # Check that the context is not in or beyond the outer wall
        if self.env_name in ['AntMazeSmall-v0', 'PointUMaze-v0']: # 0.5 margin
            if np.any(context < -1.5) or np.any(context > 9.5):
                return False
            elif np.all((np.logical_and(-2.5 < context[0], context[0] < 6.5), np.logical_and(1.5 < context[1], context[1] < 6.5))):
                return False
            else:
                return True
        elif self.env_name == "PointSpiralMaze-v0":
            if np.any(context < -9.5) or np.any(context > 9.5):
                return False
            elif np.all((np.logical_and(-2.5 < context[0], context[0] < 6.5), np.logical_and(1.5 < context[1], context[1] < 6.5))):
                return False
            elif np.all((np.logical_and(-6.5 < context[0], context[0] < -1.5), np.logical_and(-6.5 < context[1], context[1] < 6.5))):
                return False
            elif np.all((np.logical_and(-2.5 < context[0], context[0] < 10.5), np.logical_and(-6.5 < context[1], context[1] < -1.5))):
                return False
            else:
                return True
            
        elif self.env_name in ["PointNMaze-v0"]:
            if (context[0] < -1.5) or (context[0] > 9.5):
                return False
            elif (context[1] < -1.5) or (context[1] > 17.5):
                return False
            elif np.all((np.logical_and(-2.5 < context[0], context[0] < 6.5), np.logical_and(1.5 < context[1], context[1] < 6.5))):
                return False
            elif np.all((np.logical_and(1.5 < context[0], context[0] < 10.5), np.logical_and(9.5 < context[1], context[1] < 14.5))):
                return False
            else:
                return True
        
        elif self.env_name in ['sawyer_peg_pick_and_place']:
            if not np.all(np.logical_and(self.LOWER_CONTEXT_BOUNDS < context, context <self.UPPER_CONTEXT_BOUNDS)):
                return False            
            else:
                return True
        elif self.env_name in ['sawyer_peg_push']:
            if not np.all(np.logical_and(self.LOWER_CONTEXT_BOUNDS < context, context <self.UPPER_CONTEXT_BOUNDS)):
                return False            
            else:
                return True
        else:
            raise NotImplementedError

    def sample(self):
        
        sample = np.random.uniform(self.LOWER_CONTEXT_BOUNDS, self.UPPER_CONTEXT_BOUNDS)
        while not self.is_feasible(sample):
            sample = np.random.uniform(self.LOWER_CONTEXT_BOUNDS, self.UPPER_CONTEXT_BOUNDS)
        
        return sample


def get_object_states_only_from_goal(env_name, goal):
    if env_name in ['sawyer_door', 'sawyer_peg']:
        return goal[..., 4:7]

    elif env_name == 'tabletop_manipulation':
        raise NotImplementedError
    
    else:
        raise NotImplementedError

def get_original_final_goal(env_name):
    if env_name in ['AntMazeSmall-v0', 'PointUMaze-v0']:
        original_final_goal = np.array([0., 8.])
    elif env_name in ['sawyer_peg_push']:
        original_final_goal = np.array([-0.3, 0.4, 0.02])
    elif env_name in ['sawyer_peg_pick_and_place']:
        original_final_goal = np.array([-0.3, 0.4, 0.2])
    elif env_name == "PointSpiralMaze-v0":
        original_final_goal = np.array([8., -8.])
    elif env_name in ["PointNMaze-v0"]:
        original_final_goal = np.array([8., 16.])
    else:
        raise NotImplementedError
    return original_final_goal.copy()


max_episode_timesteps_dict = {'AntMazeSmall-v0' : 300,
                              'PointUMaze-v0' : 100,
                              'sawyer_peg_pick_and_place' : 200,
                              'sawyer_peg_push' : 200,
                              'PointNMaze-v0' : 100, 
                              'PointSpiralMaze-v0' : 200,
                             }

num_seed_steps_dict = { 'AntMazeSmall-v0' : 4000,
                        'PointUMaze-v0' : 2000,
                        'sawyer_peg_pick_and_place' : 2000,
                        'sawyer_peg_push' : 2000,
                        'PointNMaze-v0' : 2000, 
                        'PointSpiralMaze-v0' : 2000,
                        }

num_random_steps_dict = {'AntMazeSmall-v0' : 4000,
                         'PointUMaze-v0' : 2000,
                         'sawyer_peg_pick_and_place' : 2000,
                         'sawyer_peg_push' : 2000,
                         'PointNMaze-v0' : 2000, 
                         'PointSpiralMaze-v0' : 2000,
                        }

randomwalk_random_noise_dict = {'AntMazeSmall-v0' : 2.5,
                                'PointUMaze-v0' : 2.5,
                                'sawyer_peg_pick_and_place' : 0.1,
                                'sawyer_peg_push' : 0.1,
                                'PointNMaze-v0' : 2.5, 
                                'PointSpiralMaze-v0' : 2.5,
                                }
aim_disc_replay_buffer_capacity_dict = {'AntMazeSmall-v0' : 50000,
                                        'PointUMaze-v0' : 10000,
                                        'sawyer_peg_pick_and_place' : 30000,
                                        'sawyer_peg_push' : 30000,
                                        'PointNMaze-v0' : 10000, 
                                        'PointSpiralMaze-v0' : 20000,
                                        }



class Workspace(object):
    def __init__(self, cfg):
        self.work_dir = os.getcwd()
        print(f'workspace: {self.work_dir}')
        self.model_dir = utils.make_dir(self.work_dir, 'model')
        
        self.buffer_dir = utils.make_dir(self.work_dir, 'buffer')

        self.cfg = cfg

        self.logger = Logger(self.work_dir,
                             save_tb=cfg.log_save_tb,
                             log_frequency=cfg.log_frequency_step,
                             action_repeat=cfg.action_repeat,
                             agent='outpace_rl')

        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)

        cfg.max_episode_timesteps = max_episode_timesteps_dict[cfg.env]
        cfg.num_seed_steps = num_seed_steps_dict[cfg.env]
        cfg.num_random_steps = num_random_steps_dict[cfg.env]
        cfg.randomwalk_random_noise = randomwalk_random_noise_dict[cfg.env]
        assert cfg.aim_disc_replay_buffer_capacity == aim_disc_replay_buffer_capacity_dict[cfg.env]
     
        if cfg.env in ['sawyer_peg_push', 'sawyer_peg_pick_and_place']:
            cfg.goal_env=False
            from envs import sawyer_peg_pick_and_place, sawyer_peg_push
            if cfg.env =='sawyer_peg_pick_and_place':
                env = sawyer_peg_pick_and_place.SawyerPegPickAndPlaceV2(reward_type='sparse')
                eval_env = sawyer_peg_pick_and_place.SawyerPegPickAndPlaceV2(reward_type='sparse')
            elif cfg.env =='sawyer_peg_push':
                env = sawyer_peg_push.SawyerPegPushV2(reward_type='sparse', close_gripper=False)
                eval_env = sawyer_peg_push.SawyerPegPushV2(reward_type='sparse', close_gripper=False)
            
            from gym.wrappers.time_limit import TimeLimit
            env = TimeLimit(env, max_episode_steps=cfg.max_episode_timesteps)
            eval_env = TimeLimit(eval_env, max_episode_steps=cfg.max_episode_timesteps)
            

            if cfg.use_residual_randomwalk:
                from env_utils import ResidualGoalWrapper
                env = ResidualGoalWrapper(env, env_name = cfg.env)
                eval_env = ResidualGoalWrapper(eval_env, env_name = cfg.env)

                                       
            from env_utils import StateWrapper, DoneOnSuccessWrapper
            if cfg.done_on_success:
                relative_goal_env = False
                residual_goal_env = True if cfg.use_residual_randomwalk else False
                env = DoneOnSuccessWrapper(env, relative_goal_env = (relative_goal_env or residual_goal_env), reward_offset=0.0, earl_env = False)
                eval_env = DoneOnSuccessWrapper(eval_env, relative_goal_env = (relative_goal_env or residual_goal_env), reward_offset=0.0, earl_env = False)

            from env_utils import WraptoGoalEnv
            self.env = StateWrapper(WraptoGoalEnv(env, env_name = cfg.env))
            self.eval_env = StateWrapper(WraptoGoalEnv(eval_env, env_name = cfg.env))

            obs_spec = self.env.observation_spec()
            action_spec = self.env.action_spec()
        
        elif cfg.goal_env: # e.g. Fetch, Ant
            import gym            
            from env_utils import StateWrapper, HERGoalEnvWrapper, DoneOnSuccessWrapper, ResidualGoalWrapper             
            if cfg.env in ['AntMazeSmall-v0']:
                from gym.wrappers.time_limit import TimeLimit
                from envs.AntEnv.envs.antenv import EnvWithGoal
                from envs.AntEnv.envs.antenv.create_maze_env import create_maze_env                                              
                self.env = TimeLimit(EnvWithGoal(create_maze_env(cfg.env, cfg.seed, env_path = cfg.env_path), cfg.env), max_episode_steps=cfg.max_episode_timesteps)
                self.eval_env = TimeLimit(EnvWithGoal(create_maze_env(cfg.env, cfg.seed, env_path = cfg.env_path), cfg.env), max_episode_steps=cfg.max_episode_timesteps)
                # self.eval_env.evaluate = True # set test goal = (0,16)
                self.env.set_attribute(evaluate=False, distance_threshold=1.0, horizon=cfg.max_episode_timesteps, early_stop=False)
                self.eval_env.set_attribute(evaluate=True, distance_threshold=1.0, horizon=cfg.max_episode_timesteps, early_stop=False)


                if cfg.use_residual_randomwalk:
                    self.env = ResidualGoalWrapper(self.env, env_name = cfg.env)
                    self.eval_env = ResidualGoalWrapper(self.eval_env, env_name = cfg.env)
            elif cfg.env in ["PointUMaze-v0", "PointSpiralMaze-v0", "PointNMaze-v0"]:
                from gym.wrappers.time_limit import TimeLimit
                import mujoco_maze                                             
                self.env = TimeLimit(gym.make(cfg.env), max_episode_steps=cfg.max_episode_timesteps)
                self.eval_env = TimeLimit(gym.make(cfg.env), max_episode_steps=cfg.max_episode_timesteps)

                if cfg.use_residual_randomwalk:
                    self.env = ResidualGoalWrapper(self.env, env_name = cfg.env)
                    self.eval_env = ResidualGoalWrapper(self.eval_env, env_name = cfg.env)

            else:
                self.env = gym.make(cfg.env)
                self.eval_env = gym.make(cfg.env)

            if cfg.done_on_success:
                relative_goal_env = False
                residual_goal_env = True if cfg.use_residual_randomwalk else False
                self.env = DoneOnSuccessWrapper(self.env, relative_goal_env = (relative_goal_env or residual_goal_env))
                self.eval_env = DoneOnSuccessWrapper(self.eval_env, relative_goal_env = (relative_goal_env or residual_goal_env))
            # self.goal_env = self.env
            
            self.env= StateWrapper(HERGoalEnvWrapper(self.env, env_name= cfg.env))
            self.eval_env= StateWrapper(HERGoalEnvWrapper(self.eval_env, env_name= cfg.env))
        
                
            obs_spec = self.env.observation_spec()
            action_spec = self.env.action_spec()
            

        
        cfg.agent.action_shape = action_spec.shape
        

        
        cfg.agent.action_range = [
            float(action_spec.low.min()),
            float(action_spec.high.max())
        ]
            
        
            
        self.max_episode_timesteps = cfg.max_episode_timesteps
        
        if cfg.aim_discriminator_cfg.output_activation in [None, 'none', 'None']:
            cfg.aim_discriminator_cfg.output_activation = None
      
            

        if cfg.use_meta_nml:
            if cfg.meta_nml.num_finetuning_layers in [None, 'none', 'None']:
                cfg.meta_nml.num_finetuning_layers = None
            if cfg.meta_nml_kwargs.meta_nml_custom_embedding_key in [None, 'none', 'None']:
                cfg.meta_nml_kwargs.meta_nml_custom_embedding_key = None
            
        cfg.meta_nml.equal_pos_neg_test= cfg.meta_nml_kwargs.equal_pos_neg_test and (not cfg.meta_nml_kwargs.meta_nml_negatives_only)
        cfg.meta_nml.input_dim = self.env.goal_dim
        
        if cfg.env in ['sawyer_door', 'sawyer_peg']:      
            if cfg.aim_kwargs.aim_input_type=='default':
                cfg.aim_discriminator_cfg.x_dim = (get_object_states_only_from_goal(self.cfg.env, np.ones(self.env.goal_dim)).shape[-1])*2
            
            cfg.critic.feature_dim = self.env.obs_dim + self.env.goal_dim # [obs(ag), dg]
            cfg.actor.feature_dim = self.env.obs_dim + self.env.goal_dim # [obs(ag), dg]
            
        elif cfg.env in ['sawyer_peg_push', 'sawyer_peg_pick_and_place']:
            if cfg.aim_kwargs.aim_input_type=='default':
                cfg.aim_discriminator_cfg.x_dim = self.env.goal_dim*2
            
            cfg.critic.feature_dim = self.env.obs_dim + self.env.goal_dim # [obs(ag), dg]
            cfg.actor.feature_dim = self.env.obs_dim + self.env.goal_dim # [obs(ag), dg]
            
        else:
            if cfg.aim_kwargs.aim_input_type=='default':
                cfg.aim_discriminator_cfg.x_dim = self.env.goal_dim*2
            
            cfg.critic.feature_dim = self.env.obs_dim + self.env.goal_dim*2 # [obs, ag, dg]
            cfg.actor.feature_dim = self.env.obs_dim + self.env.goal_dim*2 # [obs, ag, dg]
            

        
        cfg.agent.goal_dim = self.env.goal_dim

        cfg.agent.obs_shape = obs_spec.shape
        # exploration agent uses intrinsic reward
        self.expl_agent = hydra.utils.instantiate(cfg.agent)
        
            
        self.expl_buffer = ReplayBuffer(obs_spec.shape, action_spec.shape,
                                        cfg.replay_buffer_capacity,
                                        self.device)
        
        self.aim_expl_buffer = ReplayBuffer(obs_spec.shape, action_spec.shape,
                                        cfg.aim_disc_replay_buffer_capacity,
                                        self.device)
        n_sampled_goal = 4
        self.randomwalk_buffer = None
        if cfg.use_residual_randomwalk:
            self.randomwalk_buffer = ReplayBuffer(obs_spec.shape, action_spec.shape,
                                        cfg.randomwalk_buffer_capacity,
                                        self.device)
            self.randomwalk_buffer = HindsightExperienceReplayWrapperVer2(self.randomwalk_buffer, 
                                                                n_sampled_goal=n_sampled_goal, 
                                                                wrapped_env=self.env,
                                                                env_name = cfg.env,
                                                                consider_done_true = cfg.done_on_success,
                                                                )

        self.goal_buffer = None


        
        self.expl_buffer = HindsightExperienceReplayWrapperVer2(self.expl_buffer, 
                                                            n_sampled_goal=n_sampled_goal, 
                                                            wrapped_env=self.env,
                                                            env_name = cfg.env,
                                                            consider_done_true = cfg.done_on_success,
                                                            )
        self.aim_expl_buffer = HindsightExperienceReplayWrapperVer2(self.aim_expl_buffer, 
                                                            n_sampled_goal=cfg.aim_n_sampled_goal, #n_sampled_goal, 
                                                            # goal_selection_strategy=KEY_TO_GOAL_STRATEGY['future'],
                                                            wrapped_env=self.env,
                                                            env_name = cfg.env,
                                                            consider_done_true = cfg.done_on_success,
                                                            )
        if cfg.use_hgg:
            from hgg.hgg import TrajectoryPool, MatchSampler            
            self.hgg_achieved_trajectory_pool = TrajectoryPool(**cfg.hgg_kwargs.trajectory_pool_kwargs)
            self.hgg_sampler = MatchSampler(goal_env=self.eval_env, 
                                            goal_eval_env = self.eval_env, 
                                            env_name=cfg.env,
                                            achieved_trajectory_pool = self.hgg_achieved_trajectory_pool,
                                            agent = self.expl_agent,
                                            **cfg.hgg_kwargs.match_sampler_kwargs
                                            )                
       
            
        self.eval_video_recorder = VideoRecorder(self.work_dir if cfg.save_video else None, dmc_env=False, env_name=cfg.env)
        self.train_video_recorder = VideoRecorder(self.work_dir if cfg.save_video else None, dmc_env=False, env_name=cfg.env)
        self.train_video_recorder.init(enabled=False)
        self.step = 0
        
        
        self.uniform_goal_sampler =  UniformFeasibleGoalSampler(env_name=cfg.env)


    def get_agent(self):                
        return self.expl_agent
        

    def get_buffer(self):                
        return self.expl_buffer
        

            
    
    def get_inv_weight_curriculum_buffer(self):        

        if self.cfg.inv_weight_curriculum_kwargs.curriculum_buffer=='aim':
            return self.aim_expl_buffer
        elif self.cfg.inv_weight_curriculum_kwargs.curriculum_buffer=='default':
            return self.expl_buffer

    def run(self):        
        self._run()
    
    def _run(self):        
        #import pdb; pdb.set_trace();
        episode, episode_reward, episode_step = 0, 0, 0
        start_time = time.time()
        
        recent_sampled_goals = Queue(self.cfg.hgg_kwargs.match_sampler_kwargs.num_episodes)

        previous_goals = None
        done = True
        info = {}
                   
        final_goal_states = np.tile(np.array([0., 8.]), (self.cfg.aim_num_precollect_init_state,1))
        final_goal_states += np.random.normal(loc=np.zeros_like(final_goal_states), scale=0.5*np.ones_like(final_goal_states))
        agent = self.get_agent()
        agent.final_goal_states = final_goal_states.copy()

        temp_obs = self.eval_env.reset()        
        recent_sampled_goals.put(self.eval_env.convert_obs_to_dict(temp_obs)['achieved_goal'].copy())


        current_pocket_success = 0
        current_pocket_trial = 0
        while self.step <= self.cfg.num_train_steps:
            #print(self.step, self.hgg_sampler.sample(np.random.randint(len(self.hgg_sampler.pool))).copy())
            if done:
                #print(self.step, self.max_episode_timesteps, self.cfg.hgg_kwargs.hgg_sampler_update_frequency, episode)
                
                if self.step > 0:
                    current_pocket_trial += 1
                    if info['is_success']:
                        current_pocket_success += 1 
                    
                    # hgg update
                    if self.cfg.use_hgg :
                        
                        if episode % self.cfg.hgg_kwargs.hgg_sampler_update_frequency ==0 :    
                            #import pdb; pdb.set_trace();                        
                            initial_goals = []
                            desired_goals = []                      
                            for i in range(self.cfg.hgg_kwargs.match_sampler_kwargs.num_episodes):                                
                                temp_obs = self.eval_env.convert_obs_to_dict(self.eval_env.reset())
                                goal_a = temp_obs['achieved_goal'].copy()                                
                                noise_scale = 0.5                    
                                noise = np.random.normal(loc=np.zeros_like(goal_a), scale=noise_scale*np.ones_like(goal_a))
                                goal_d = goal_a + noise # These will be meaningless after achieved_goals are accumulated in hgg_achieved_trajectory_pool
                                initial_goals.append(goal_a.copy())
                                desired_goals.append(goal_d.copy()) 
                            hgg_start_time = time.time()
                            hgg_sampler = self.hgg_sampler
                            #import pdb; pdb.set_trace();
                            hgg_sampler.update(initial_goals, desired_goals, replay_buffer = self.expl_buffer, meta_nml_epoch=episode)           
                                 
                hgg_sampler = self.hgg_sampler
                n_iter = 0
                while True:
                    # print('hgg sampler pool len : {} step : {}'.format(len(hgg_sampler.pool), self.step))
                    sampled_goal = hgg_sampler.sample(np.random.randint(len(hgg_sampler.pool))).copy()                        
                    obs = self.env.reset(goal = sampled_goal)

                    if not self.env.is_successful(obs):
                        break
                    n_iter +=1
                    if n_iter==10:
                        break

                if recent_sampled_goals.full():
                    recent_sampled_goals.get()
                recent_sampled_goals.put(sampled_goal)
                assert (sampled_goal == self.env.goal.copy()).all()
                
                final_goal = self.env.goal.copy()                

            
                episode_reward = 0
                episode_step = 0
                episode += 1
                episode_observes = [obs]
            # import pdb; pdb.set_trace();
            agent = self.get_agent()
            replay_buffer = self.get_buffer()
            

            # sample action for data collection
            if self.step < self.cfg.num_random_steps or (self.cfg.randomwalk_method == 'rand_action' and self.env.is_residual_goal):
                spec = self.env.action_spec()                
                action = np.random.uniform(spec.low, spec.high,
                                        spec.shape)
                
            else: 
                with utils.eval_mode(agent):
                    action = agent.act(obs, spec = self.env.action_spec(), sample=True)
            
            #import pdb; pdb.set_trace();
            next_obs, reward, done, info = self.env.step(action)
            #if(done):
            #    import pdb; pdb.set_trace();
            # if(done == False and info.get('is_current_goal_success')==True):
            #     import pdb; pdb.set_trace();
            
            episode_reward += reward
            episode_observes.append(next_obs)

            last_timestep = True if (episode_step+1) % self.max_episode_timesteps == 0 or done else False

            replay_buffer.add(obs, action, reward, next_obs, info.get('is_current_goal_success'), last_timestep)
            self.aim_expl_buffer.add(obs, action, reward, next_obs, info.get('is_current_goal_success'), last_timestep)
                
            if last_timestep:
                #import pdb; pdb.set_trace();
                replay_buffer.add_trajectory(episode_observes)
                replay_buffer.store_episode()
                self.aim_expl_buffer.store_episode()
                if self.randomwalk_buffer is not None:
                    self.randomwalk_buffer.store_episode()
                if self.randomwalk_buffer is not None:
                    if (not replay_buffer.full) and (not self.randomwalk_buffer.full):
                        assert self.step+1 == self.randomwalk_buffer.idx + replay_buffer.idx
                else:
                    if not replay_buffer.full:
                        assert self.step+1 == replay_buffer.idx

                if self.cfg.use_hgg:                    
                    temp_episode_observes = copy.deepcopy(episode_observes)
                    temp_episode_ag = []                                        
                     # NOTE : should it be [obs, ag] ?
                    if 'aim_f' in self.hgg_sampler.cost_type or 'meta_nml' in self.hgg_sampler.cost_type:
                        temp_episode_init = self.eval_env.convert_obs_to_dict(temp_episode_observes[0])['achieved_goal'] # for bias computing
                    else:    
                        raise NotImplementedError
                        

                    for k in range(len(temp_episode_observes)):
                        temp_episode_ag.append(self.eval_env.convert_obs_to_dict(temp_episode_observes[k])['achieved_goal'])
                    
                    if getattr(self.env, 'full_state_goal', False):
                        raise NotImplementedError("You should modify the code when full_state_goal (should address achieved_goal to compute goal distance below)")


                    achieved_trajectories = [np.array(temp_episode_ag)] # list of [ts, dim]
                    achieved_init_states = [temp_episode_init] # list of [ts(1), dim]

                    selection_trajectory_idx = {}
                    #for i in range(len(achieved_trajectories)):                                                 
                    threshold = 0.2
                    if goal_distance(achieved_trajectories[0][0], achieved_trajectories[0][-1])>threshold: # if there is a difference btw first and last timestep ?
                        selection_trajectory_idx[0] = True
                    
                    hgg_achieved_trajectory_pool = self.hgg_achieved_trajectory_pool
                    for idx in selection_trajectory_idx.keys():
                        hgg_achieved_trajectory_pool.insert(achieved_trajectories[idx].copy(), achieved_init_states[idx].copy())
                    
            obs = next_obs
            episode_step += 1
            self.step += 1
            
            if self.cfg.use_residual_randomwalk:
                if info.get('is_current_goal_success'): #succeed original goal
                    self.env.original_goal_success = True
                    noise = np.random.uniform(low=-self.cfg.randomwalk_random_noise, high=self.cfg.randomwalk_random_noise, size=self.env.goal_dim)
                    residual_goal = self.env.convert_obs_to_dict(obs)['achieved_goal'] + noise
                    self.env.reset_goal(residual_goal)
                    obs[-self.env.goal_dim:] = residual_goal.copy()
                if (episode_step) % self.max_episode_timesteps == 0: #done only horizon ends
                    done = True
                    info['is_success'] = self.env.original_goal_success


                    

@hydra.main(config_path='./config', config_name='config_outpace.yaml')
def main(cfg):
    import os
    os.environ['HYDRA_FULL_ERROR'] = str(1)
    from outpace_train import Workspace as W
    

    workspace = W(cfg)
    workspace.run()


if __name__ == '__main__':
    main()
