import numpy as np
import tensorflow as tf

import dm_soccer2gym
from spinup.utils.test_policy import load_policy_and_env

num_trials = 10
render = True


def demo_1v0(model_path, model_num):
    """
    Function that runs the 1v0 policy, from a given path, and 
    model number.
    """
    env = dm_soccer2gym.make("1vs0", task_kwargs={'time_limit': 30., 'disable_jump': True, \
                                                  'dist_thresh': .03, 'control_timestep': 0.05})
    _, get_action = load_policy_and_env(model_path, model_num)
    
    for k in range(num_trials):
        obs=env.reset()
        d=False
        while not d:
            a = [get_action(o) for o in obs]
            obs, r, d, _ = env.step(a)
            if render: env.render()
            
    tf.reset_default_graph()


def demo_1v0_versus_1v1(model_path_1v0, model_num_1v0, model_path_1v1, model_num_1v1):
    """
    Function that runs the 1v0 policy, from a given path, and 
    model number, against the 1v1 policy, from a given path, and 
    model number.
    """
    env = dm_soccer2gym.make("1vs1", task_kwargs={'time_limit': 30., 'disable_jump': True, \
                                                  'dist_thresh': .03, 'control_timestep': 0.05})
    _, get_action_1 = load_policy_and_env(model_path_1v0, model_num_1v0)
    
    g = tf.Graph()
    with g.as_default():
        _, get_action_2 = load_policy_and_env(model_path_1v1, model_num_1v1)

    for k in range(num_trials):
        obs=env.reset()
        d=False
        while not d:
            a = [get_action_1(obs[0][:18]), get_action_2(obs[1])]
            obs, r, d, _ = env.step(a)
            if render: env.render()
            
    tf.reset_default_graph()
            
            
def demo_1v1(model_path_1v1_1, model_num_1v1_1, model_path_1v1_2, model_num_1v1_2):
    """
    Function that runs two 1v1 policies, from given paths, and 
    model numbers.
    """
    env = dm_soccer2gym.make("1vs1", task_kwargs={'time_limit': 30., 'disable_jump': True, 
                                                  'dist_thresh': .03, 'control_timestep': 0.05})
    _, get_action_1 = load_policy_and_env(model_path_1v1_1, model_num_1v1_1)
    
    g = tf.Graph()
    with g.as_default():
        _, get_action_2 = load_policy_and_env(model_path_1v1_2, model_num_1v1_2)

    for k in range(num_trials):
        obs=env.reset()
        d=False
        while not d:
            a = [get_action_1(obs[0]), get_action_2(obs[1])]
            obs, r, d, _ = env.step(a)
            if render: env.render()
            
    tf.reset_default_graph()

def demo_1v1_versus_2v2(model_path_1v1_1, model_num_1v1_1, model_path_1v1_2, model_num_1v1_2, model_path_2v2, model_num_2v2):
    """
    Function that runs two 1v1 policies, from given paths, and 
    model numbers, against the 2v2 policy, from a given path, and 
    model number.
    """
    env = dm_soccer2gym.make('2vs2', task_kwargs={"time_limit": 45., "disable_jump": True, \
                                                  "dist_thresh": 0.03, 'control_timestep': 0.05})
    _, get_action = load_policy_and_env(model_path_2v2, model_num_2v2, two_p=True)

    g2 = tf.Graph()
    with g2.as_default():
        _, get_action_1 = load_policy_and_env(model_path_1v1_1, model_num_1v1_1)

    g3 = tf.Graph()
    with g3.as_default():
        _, get_action_2 = load_policy_and_env(model_path_1v1_2, model_num_1v1_2)

    for k in range(num_trials):
        obs=env.reset()
        d=False
        while not d:
            a = [get_action_1(obs[0][np.r_[0:18, 18:24]]), get_action_2(obs[1][np.r_[0:18, 30:36]]), *get_action(*obs[2:4])]
            obs, r, d, _ = env.step(a)
            if render: env.render()
            
    tf.reset_default_graph()
            

def demo_2v2(model_path_2v2_1, model_num_2v2_1, model_path_2v2_2, model_num_2v2_2):
    """
    Function that runs two 2v2 policies, from given paths, and 
    model numbers.
    """
    env = dm_soccer2gym.make('2vs2', task_kwargs={"time_limit": 45., "disable_jump": True, 
                                                  "dist_thresh": 0.03, 'control_timestep': 0.05})
    _, get_action_1 = load_policy_and_env(model_path_2v2_1, model_num_2v2_1, two_p=True)
    
    g = tf.Graph()
    with g.as_default():
        _, get_action_2 = load_policy_and_env(model_path_2v2_2, model_num_2v2_2, two_p=True)

    for k in range(num_trials):
        obs=env.reset()
        d=False
        while not d:
            a = [*get_action_1(*obs[0:2]), *get_action_2(*obs[2:4])]
            obs, r, d, _ = env.step(a)
            if render: env.render()
            
    tf.reset_default_graph()
            

if __name__ == "__main__":
    print("Running 1v0 (DR) policy")
    demo_1v0("models/TD3/1vs0/2020-09-12_23-40-25_td3_soccer_1vs0_dense_0.05", 8299999)
    
    print("Running 1v0 (DR) versus 1v1 (DR + ES) policy")
    demo_1v0_versus_1v1("models/TD3/1vs0/2020-09-12_23-40-25_td3_soccer_1vs0_dense_0.05", 8299999,
                        "models/TD3/1vs1/2020-10-08_23-07-33_td3_soccer_1vs1_dense_0.05", 8629999)
                        
    print("Running 1v1 (DR + ES) versus 1v1 (DR + ES) policy")
    demo_1v1("models/TD3/1vs1/2020-10-08_23-06-32_td3_soccer_1vs1_dense_0.05", 9389999,
             "models/TD3/1vs1/2020-10-08_23-07-33_td3_soccer_1vs1_dense_0.05", 8629999)
          
    print("Running 1v1 (DR + ES) versus 2v2 (DR + ES) policy")   
    demo_1v1_versus_2v2("models/TD3/1vs1/2020-10-08_23-06-32_td3_soccer_1vs1_dense_0.05", 9389999,
                        "models/TD3/1vs1/2020-10-08_23-07-33_td3_soccer_1vs1_dense_0.05", 8629999,
                        "models/TD3/2vs2/2021-03-29_13-37-57_td3_soccer_2vs2_sep_policy_es_dense_0.05", 12339999)
    
    print("Running 2v2 (DR + ES) versus 2v2 (DR + ES + HCT) policy")                   
    demo_2v2("models/TD3/2vs2/2021-03-29_13-37-57_td3_soccer_2vs2_sep_policy_es_dense_0.05", 12339999,
             "models/TD3/2vs2/2021-02-04_16-25-36_td3_soccer_2vs2_sep_policy_es_dense_0.1", 14559999)
            
    print("Running 2v2 (DR + ES + HCT) versus 2v2 (DR + ES + HCT) policy") 
    demo_2v2("models/TD3/2vs2/2021-02-04_16-25-36_td3_soccer_2vs2_sep_policy_es_dense_0.1", 14559999,
             "models/TD3/2vs2/2021-03-29_13-38-05_td3_soccer_2vs2_sep_policy_es_dense_0.1", 14339999)

