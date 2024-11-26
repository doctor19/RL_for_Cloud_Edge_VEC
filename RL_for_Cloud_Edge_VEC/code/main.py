from tensorflow.keras.optimizers import Adam
import random
import copy
import json
import timeit
import warnings
from tempfile import mkdtemp
import gym
import numpy as np
import pandas as pd
from gym import spaces
from gym.utils import seeding
from rl.agents.ddpg import DDPGAgent
#from rl.agents.dqn import DQNAgent
from rl.agents.sarsa import SARSAAgent
from rl.callbacks import Callback, FileLogger, ModelIntervalCheckpoint
from rl.memory import SequentialMemory
#from rl.policy import EpsGreedyQPolicy
from rl.random import OrnsteinUhlenbeckProcess
from tensorflow.keras.backend import cast
from tensorflow.keras.layers import (Activation, Concatenate, Dense, Dropout,
                                     Flatten, Input)
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
import sys

from fuzzy_controller import *
from environment import *
from mix_state_env import MixStateEnv
from model import *
from policy import *
from callback import *
from fuzzy_controller import *
import os
from config import *
from MyGlobal import MyGlobals
from keras.models import load_model

from dqnMEC import DQNAgent
from siblingDQN import SiblingDQN
from ExpectedTaskDQN import ExpectedTaskDQN

os.environ["CUDA_VISIBLE_DEVICES"]="-1"

def Run_Random(folder_name):
    MyGlobals.folder_name = folder_name + '/'
    env = BusEnv("Random")
    env.seed(123)
    observation = None
    done = False
    actions = [0, 1, 2, 3, 4, 5]
    for i in range(NB_STEPS):
        if observation is None:
            try:
                env.reset()
            except Exception as e:
                print(e)
        # Determine the percentage of offload to server
        action = random.choices(actions, weights=(4, 1, 1, 1, 1, 1))[0]
        observation, r, done, info = env.step(action)
        if done:
            done = False
            try:
                env.reset()
            except Exception as e:
                print(e)
                
def runShortestLatencyGreedy(folder_name):
    MyGlobals.folder_name = folder_name + '/'
    env = BusEnv("Random")
    env.seed(123)
    observation = None
    done = False
    actions = [range(NUM_ACTION)]
    for i in range(NB_STEPS):
        if observation is None:
            try:
                env.reset()
            except Exception as e:
                print("runShortestLatencyGreedy: ", e)
        # Determine the percentage of offload to server
        min_est_latency = 100
        action = 0
        for server in range(NUM_ACTION):
            time_before_return, _ = env.estimate(server)
            if (min_est_latency > time_before_return):
                min_est_latency = time_before_return
                action = server
        
        observation, r, done, info = env.step(action)
        if done:
            done = False
            try:
                env.reset()
            except Exception as e:
                print(e)
    
def runShortestExtraTimeGreedy(folder_name):
    MyGlobals.folder_name = folder_name + '/'
    env = BusEnv("Random")
    env.seed(123)
    observation = None
    done = False
    actions = [range(NUM_ACTION)]
    for i in range(NB_STEPS):
        if observation is None:
            try:
                env.reset()
            except Exception as e:
                print("runShortestExtraTimeGreedy: ", e)
        # Determine the percentage of offload to server
        max_est_extra_time = -100
        action = 0
        for server in range(NUM_ACTION):
            time_before_return, est_observation = env.estimate(server)
            est_extra_time = min(0, est_observation[-1] - time_before_return)
            if (max_est_extra_time < est_extra_time):
                max_est_extra_time = est_extra_time
                action = server
        
        observation, r, done, info = env.step(action)
        if done:
            done = False
            try:
                env.reset()
            except Exception as e:
                print(e)
            

#using for DQL
# def build_model(state_size, num_actions):
#     input = Input(shape=(1,state_size))
#     x = Flatten()(input)
#     #x = Dense(16, activation='relu')(x)

#     x = Dense(32, activation='relu')(x)

#     x = Dense(32, activation='relu')(x)
  
#     x = Dense(16, activation='relu')(x)

#     output = Dense(num_actions, activation='linear')(x)
#     model = Model(inputs=input, outputs=output)
#     return model

def build_model(state_size, num_actions):
    input = Input(shape=(1,state_size))
    x = Flatten()(input)

    x = Dense(32, activation='relu', kernel_initializer='he_uniform')(x)
    
    #x = Dense(64, activation='relu', kernel_initializer='he_uniform')(x)

    x = Dense(64, activation='relu', kernel_initializer='he_uniform')(x)
  
    x = Dense(32, activation='relu', kernel_initializer='he_uniform')(x)

    output = Dense(num_actions, activation='linear')(x)
    model = Model(inputs=input, outputs=output)
    return model

def get_model():
    try:
        model = load_model('my_model.h5')
    except Exception as e: 
        print("Error in get_model")
        print(e)
    return model

def Run_DQL(folder_name):
    model=build_model(NUM_STATE, NUM_ACTION)
    #model.save('my_model.h5')
    #model = load_model('my_model.h5')
    num_actions = NUM_ACTION
    #policy = EpsLinearDecreaseQPolicy(maxeps = 1, mineps = 0, subtrahend = 0.00001)
    policy = EpsGreedyHardDecreasedQPolicy(eps=.2, decreased_quantity=.1,nb_hard_decreased_steps=62500)
    #policy = EpsGreedyQPolicy(0.1)
    MyGlobals.folder_name = folder_name + '/'
    env = MixStateEnv()
    env.seed(123)
    memory = SequentialMemory(limit=5000, window_length=1)
    
    dqn = DQNAgent(model=model, nb_actions=num_actions, memory=memory, nb_steps_warmup=10,\
              batch_size = 32, target_model_update=1e-3, policy=policy,gamma=0.95,train_interval=5)
    #files = open("testDQL.csv","w")
    #files.write("kq\n")
    #create callback
    callbacks = CustomerTrainEpisodeLogger("DQL_5phut.csv")
    callback2 = ModelIntervalCheckpoint("weight_DQL.h5f",interval=50000)
    #callback3 = TestLogger11(files)
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])
    try:
        dqn.fit(env, nb_steps= NB_STEPS, visualize=False, verbose=2,callbacks=[callbacks,callback2])
        # dqn.policy = EpsGreedyQPolicy(0.0)
        # dqn.test(env, nb_episodes = 30)
    except Exception as e:
        print(e)

def Run_ExpectedTaskDQN(folder_name):
    model=build_model(NUM_STATE, NUM_ACTION)
    #model = load_model('my_model.h5')
    num_actions = NUM_ACTION
    #policy = EpsLinearDecreaseQPolicy(maxeps = 1, mineps = 0, subtrahend = 0.00001)
    policy = EpsGreedyHardDecreasedQPolicy(eps=.2, decreased_quantity=.1,nb_hard_decreased_steps=62500)
    #policy = EpsGreedyQPolicy(0.1)
    MyGlobals.folder_name = folder_name + '/'
    env = MixStateEnv()
    env.seed(123)
    memory = SequentialMemory(limit=5000, window_length=1)
    
    dqn = ExpectedTaskDQN(model=model, nb_actions=num_actions, memory=memory, nb_steps_warmup=10,\
              batch_size = 32, target_model_update=1e-3, policy=policy,gamma=0.95,train_interval=5)
    #files = open("testDQL.csv","w")
    #files.write("kq\n")
    #create callback
    callbacks = CustomerTrainEpisodeLogger("DQL_5phut.csv")
    callback2 = ModelIntervalCheckpoint("weight_DQL.h5f",interval=50000)
    #callback3 = TestLogger11(files)
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])
    try:
        dqn.fit(env, nb_steps= NB_STEPS, visualize=False, verbose=2,callbacks=[callbacks,callback2])
        # dqn.policy = EpsGreedyQPolicy(0.0)
        # dqn.test(env, nb_episodes = 30)
    except Exception as e:
        print(e)
        
def Run_ExpectedTaskDDQN(folder_name):
    model=build_model(NUM_STATE, NUM_ACTION)
    #model = load_model('my_model.h5')
    num_actions = NUM_ACTION
    #policy = EpsLinearDecreaseQPolicy(maxeps = 1, mineps = 0, subtrahend = 0.00001)
    policy = EpsGreedyHardDecreasedQPolicy(eps=.2, decreased_quantity=.1,nb_hard_decreased_steps=62500)
    #policy = EpsGreedyQPolicy(0.1)
    MyGlobals.folder_name = folder_name + '/'
    env = MixStateEnv()
    env.seed(123)
    memory = SequentialMemory(limit=5000, window_length=1)
    
    dqn = ExpectedTaskDQN(model=model, nb_actions=num_actions, memory=memory, nb_steps_warmup=10,\
              batch_size = 32, target_model_update=1e-3, policy=policy,gamma=0.95,train_interval=4,
              enable_double_dqn=True)
    #files = open("testDQL.csv","w")
    #files.write("kq\n")
    #create callback
    callbacks = CustomerTrainEpisodeLogger("ExpectedTaskDDQN_5phut.csv")
    callback2 = ModelIntervalCheckpoint("weight_ExpectedTaskDDQN.h5f",interval=50000)
    #callback3 = TestLogger11(files)
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])
    try:
        dqn.fit(env, nb_steps= NB_STEPS, visualize=False, verbose=2,callbacks=[callbacks,callback2])
        # dqn.policy = EpsGreedyQPolicy(0.0)
        # dqn.test(env, nb_episodes = 30)
    except Exception as e:
        print(e)
        
def Run_DDQL(folder_name):
    model=build_model(NUM_STATE, NUM_ACTION)
    #model = load_model('my_model.h5')
    num_actions = NUM_ACTION
    #policy = EpsLinearDecreaseQPolicy(maxeps = 1, mineps = 0, subtrahend = 0.00001)
    policy = EpsGreedyHardDecreasedQPolicy(eps=.2, decreased_quantity=.1,nb_hard_decreased_steps=62500)
    #policy = EpsGreedyQPolicy(0.1)
    MyGlobals.folder_name = folder_name + '/'
    env = MixStateEnv()
    env.seed(123)
    memory = SequentialMemory(limit=5000, window_length=1)
    
    dqn = DQNAgent(model=model, nb_actions=num_actions, memory=memory, nb_steps_warmup=10,\
              batch_size = 32, target_model_update=1e-3, policy=policy,gamma=0.95,train_interval=5, 
              enable_double_dqn=True)
    #files = open("testDQL.csv","w")
    #files.write("kq\n")
    #create callback
    callbacks = CustomerTrainEpisodeLogger("DDQL_5phut.csv")
    callback2 = ModelIntervalCheckpoint("weight_DDQL.h5f",interval=50000)
    #callback3 = TestLogger11(files)
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])
    try:
        dqn.fit(env, nb_steps= NB_STEPS, visualize=False, verbose=2,callbacks=[callbacks,callback2])
        # dqn.policy = EpsGreedyQPolicy(0.0)
        # dqn.test(env, nb_episodes = 30)
    except Exception as e:
        print(e)
        
def Run_DuelingDQL(folder_name):
    model=build_model(NUM_STATE, NUM_ACTION)
    #model = load_model('my_model.h5')
    num_actions = NUM_ACTION
    #policy = EpsLinearDecreaseQPolicy(maxeps = 1, mineps = 0, subtrahend = 0.00001)
    policy = EpsGreedyHardDecreasedQPolicy(eps=.2, decreased_quantity=.1,nb_hard_decreased_steps=62500)
    #policy = EpsGreedyQPolicy(0.1)
    MyGlobals.folder_name = folder_name + '/'
    env = MixStateEnv()
    env.seed(123)
    memory = SequentialMemory(limit=5000, window_length=1)
    
    dqn = DQNAgent(model=model, nb_actions=num_actions, memory=memory, nb_steps_warmup=10,\
              batch_size = 32, target_model_update=1e-3, policy=policy,gamma=0.95,train_interval=5, 
              enable_dueling_network=True)
    #files = open("testDQL.csv","w")
    #files.write("kq\n")
    #create callback
    callbacks = CustomerTrainEpisodeLogger("DuelingDQL_5phut.csv")
    callback2 = ModelIntervalCheckpoint("weight_DuelingDQL.h5f",interval=50000)
    #callback3 = TestLogger11(files)
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])
    try:
        dqn.fit(env, nb_steps= NB_STEPS, visualize=False, verbose=2,callbacks=[callbacks,callback2])
        # dqn.policy = EpsGreedyQPolicy(0.0)
        # dqn.test(env, nb_episodes = 30)
    except Exception as e:
        print(e)
        
def runOtherExpectedTaskDDQN(folder_name):
    model=build_model(NUM_STATE, NUM_ACTION)
    #model = load_model('my_model.h5')
    num_actions = NUM_ACTION
    #policy = EpsLinearDecreaseQPolicy(maxeps = 1, mineps = 0, subtrahend = 0.00001)
    policy = EpsGreedyHardDecreasedQPolicy(eps=.2, decreased_quantity=.1,nb_hard_decreased_steps=25000)
    #policy = EpsGreedyQPolicy(0.1)
    MyGlobals.folder_name = folder_name + '/'
    env = NoFogEnv()
    env.seed(123)
    memory = SequentialMemory(limit=5000, window_length=1)
    
    dqn = ExpectedTaskDQN(model=model, nb_actions=num_actions, memory=memory, nb_steps_warmup=10,\
              batch_size = 32, target_model_update=1e-3, policy=policy,gamma=0.95,train_interval=2,
              enable_double_dqn=True)
    #files = open("testDQL.csv","w")
    #files.write("kq\n")
    #create callback
    callbacks = CustomerTrainEpisodeLogger("ExpectedTaskDDQN_5phut.csv")
    callback2 = ModelIntervalCheckpoint("weight_ExpectedTaskDDQN.h5f",interval=50000)
    #callback3 = TestLogger11(files)
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])
    try:
        dqn.fit(env, nb_steps= NB_STEPS, visualize=False, verbose=2,callbacks=[callbacks,callback2])
        # dqn.policy = EpsGreedyQPolicy(0.0)
        # dqn.test(env, nb_episodes = 30)
    except Exception as e:
        print(e)

if __name__=="__main__":
    for i in range(2, 3):
        #Run_DQL("DQN_0.2_0.1_25/" + str(6))
        #Run_ExpectedTaskDQN("Test" + str(i))
        #Run_ExpectedTaskDDQN("ExpectedTaskDDQN_0.2_0.1_25_2000_tasks/" + str(2))
        #Run_ExpectedTaskDDQN("ExpectedTaskDDQN_0.2_0.1_25_normal_no_cloud/" + str(i))
        #Run_DDQL("DDQN_0.2_0.1_25_2000_tasks/" + str(1))
        #Run_DuelingDQL("DuelingDQN_0.2_0.1_25/" + str(i))
        #Run_ExpectedTaskDQN("ExpectedTaskDQN_decreased_1_downto_0_0.00001/" + str(i))
        #runShortestLatencyGreedy("ShortestLatencyGreedy_5000_tasks/2")
        #runShortestExtraTimeGreedy("ShortestExtraTimeGreedy_normal_no_cloud/1")
        runOtherExpectedTaskDDQN("a/" + str(11))
        #runOtherExpectedTaskDDQN("ExpectedTaskDDQN_0.2_0.1_25_normal_no_fog_servers/" + str(i))
    #Run_DQL("a/7")
    #runShortestLatencyGreedy("ShortestLatencyGreedy/1")
    #runShortestExtraTimeGreedy("a/2")
    #Run_DDQL("DDQN7")
    #Run_DDQL("DDQN2_no_energy"+str(NUM_ACTION - 1)+"VS")
    #Run_DuelingDQL("DuelingDQN4_"+str(NUM_ACTION - 1)+"VS")
    #Run_DoubleDuelingDQL("DoubleDuelingDQN1")
    #Run_Random("Random_4_1_1_1_1_1")

















