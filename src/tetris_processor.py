from rl.core import Processor
from src.configuration import Num_Types
import numpy as np


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

