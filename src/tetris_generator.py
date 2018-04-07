from copy import deepcopy

from keras.utils import Sequence
from src.Tetris_Env import TetrisEnv
from src.tetris_processor import TetrisProcessor
import numpy as np

class TetrisGenerator(Sequence):

    env = TetrisEnv()
    action_space = env.action_space
    processor = TetrisProcessor()
    counter = 0
    observation = None

    def __init__(self):
        self.observation = self.env.reset()
        self.counter = 1

    def __getitem__(self, index):
        self.counter = self.counter + 1
        if self.counter > 100:
            self.__init__()
        best_evaluation = - 100000
        best_action_num = -1
        action_num = 0
        for orient, slot in self.action_space.legal_moves[self.env.currentPiece]:
            board_copy = deepcopy(self.env.board)
            top_copy = deepcopy(self.env.top)
            evaluation, _, _ = self.env.perform_action(board_copy, top_copy, orient, slot)
            if evaluation > best_evaluation:
                best_evaluation = evaluation
                best_action_num = action_num
            action_num = action_num + 1
        processed_observation = self.processor.process_observation(self.observation)
        processed_action = self.process_action(best_action_num)
        data_item = (np.expand_dims(np.expand_dims(processed_observation,0),0), np.expand_dims(processed_action,0))
        self.observation, _, _, _ = self.env.step(best_action_num)
        return data_item

    def process_action(self, action):
        processed_action = np.zeros(self.action_space.n)
        processed_action[action] = 1
        return processed_action

    def __len__(self):
        return 1300000
