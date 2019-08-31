import pytest
import numpy as np

from stable_baselines import DQN
from stable_baselines.common import set_global_seeds

# Using the Deep Q-Network
MODEL_LIST_DISCRETE = [DQN]

@pytest.mark.parametrize("model_class", MODEL_LIST_DISCRETE)
def test_perf_cartpole(model_class):

    # https://towardsdatascience.com/stable-baselines-a-fork-of-openai-baselines-reinforcement-learning-made-easy-df87c4b2fc82
    # Log using the tensorboard
    model = model_class(policy="MlpPolicy", env='CartPole-v1', tensorboard_log="/tmp/log/final/cartpole")
    model.learn(total_timesteps=int(1e5), seed=0)
    #Save the model
    model.save('cartpole')
    #Get the cartpole environment
    env = model.get_env()
    #set the seed value
    set_global_seeds(0)
    #Reset the environment
    obs = env.reset()
    while True:
        #Predict
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        #Render the frames
        env.render()