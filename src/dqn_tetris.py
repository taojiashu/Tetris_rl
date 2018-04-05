#from __future__ import division

import argparse

import numpy as np
from keras.layers import Dense, Activation, Flatten
from keras.models import Sequential
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.callbacks import FileLogger, ModelIntervalCheckpoint
from rl.core import Processor
from rl.memory import SequentialMemory
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy

from src.Tetris_Env import TetrisEnv
from src.configuration import Num_Types

Window_Length = 4


class TetrisProcessor(Processor):

    def process_observation(self, observation):

        board, piece1, piece2 = observation
        flatten_board = np.array(board).flatten()
        list_1= []
        for i in range(Num_Types):
            if i == piece1:
                list_1.append(1)
            else:
                list_1.append(0)
        for i in range(Num_Types):
            if i == piece2:
                list_1.append(1)
            else:
                list_1.append(0)

        return np.append(flatten_board, list_1).astype('uint8')  # saves storage in experience memory



    def process_state_batch(self, batch):

        # We could perform this processing step in `process_observation`. In this case, however,
        # we would need to store a `float32` array instead, which is 4x more memory intensive than
        # an `uint8` array. This matters if we store 1M observations.
        processed_batch = batch.astype('float32') / 255.
        return processed_batch

    def process_reward(self, reward):
        return reward


parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['train', 'test'], default='train')
parser.add_argument('--weights', type=str, default=None)

args = parser.parse_args()



# Get the environment and extract the number of actions.
env = TetrisEnv()
np.random.seed(123)
env.seed(123)

nb_actions = env.action_space.n

model = Sequential()
input_shape = (Window_Length,) + (224,)

model.add(Dense(128, input_shape = input_shape))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
print(model.summary())



# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and

# even the metrics!

memory = SequentialMemory(limit=1000000, window_length=4)

processor = TetrisProcessor()

# Select a policy. We use eps-greedy action selection, which means that a random action is selected
# with probability eps. We anneal eps from 1.0 to 0.1 over the course of 1M steps. This is done so that
# the agent initially explores the environment (high eps) and then gradually sticks to what it knows
# (low eps). We also set a dedicated eps value that is used during testing. Note that we set it to 0.05
# so that the agent still performs some random actions. This ensures that the agent cannot get stuck.
policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05,
                              nb_steps=1000000)

# The trade-off between exploration and exploitation is difficult and an on-going research topic.
# If you want, you can experiment with the parameters or use a different policy. Another popular one
# is Boltzmann-style exploration:
# policy = BoltzmannQPolicy(tau=1.)
# Feel free to give it a try!

dqn = DQNAgent(model=model, nb_actions=nb_actions, policy=policy, memory=memory,
               processor=processor, nb_steps_warmup=50000, gamma=.99, target_model_update=10000,
               train_interval=4, delta_clip=1.)
dqn.compile(Adam(lr=.00025), metrics=['mae'])

if args.mode == 'train':
    # Okay, now it's time to learn something! We capture the interrupt exception so that training
    # can be prematurely aborted. Notice that you can the built-in Keras callbacks!
    weights_filename = 'dqn_tetris_weights.h5f'
    checkpoint_weights_filename = 'dqn_tetris' + '_weights_{step}.h5f'
    log_filename = 'dqn_tetris_log.json'
    callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=250000)]
    callbacks += [FileLogger(log_filename, interval=100)]
    dqn.fit(env, callbacks=callbacks, nb_steps=1750000, log_interval=10000)
    # After training is done, we save the final weights one more time.
    dqn.save_weights(weights_filename, overwrite=True)
    # Finally, evaluate our algorithm for 10 episodes.
    dqn.test(env, nb_episodes=10, visualize=False)
elif args.mode == 'test':
    weights_filename = 'dqn_tetris_weights.h5f'
    if args.weights:
        weights_filename = args.weights
    dqn.load_weights(weights_filename)
    dqn.test(env, nb_episodes=10, visualize=True)