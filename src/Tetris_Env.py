from rl.core import Env, Space
from copy import deepcopy
from src.configuration import pOrients, pWidth, pHeight, pTop, pBottom, Num_Types, Col, Row
from random import Random
import numpy as np


class TetrisEnv(Env):
    randomness = Random()
    randomness.seed = 0

    board = None
    top = None
    currentPiece = None
    nextPiece = None
    info = {"dummp":1}
    total_score = 0
    evaluation = 0

    def __init__(self):
        self.board = [[0] * Col for i in range(Row)]
        self.top = [0] * Col
        self.currentPiece = self.new_piece()
        self.nextPiece = self.new_piece()
        self.action_space = self.ActionSpace(self)

    def step(self, action):
        max_evaluation = self.find_max_evaluation()
        orient, slot = self.action_space.legal_moves[self.currentPiece][action % len(
            self.action_space.legal_moves[self.currentPiece])]
        score, is_done = self.perform_action(self.board, self.top, orient, slot)

        ## self.total_score = self.total_score + score
        ## new_evaluation = self.evaluate_board(self.board, self.top) + self.total_score * 0.760666
        ## reward = new_evaluation - self.evaluation + 100
        ## self.evaluation = new_evaluation
        reward = 0
        if self.evaluate_board(self.board, self.top) >= max_evaluation - 0.0001:
            reward = 100

        self.currentPiece = self.nextPiece
        self.nextPiece = self.new_piece()
        observation = (deepcopy(self.board), self.currentPiece, self.nextPiece)
        return observation, reward, is_done, self.info

    def find_max_evaluation(self):
        max_evaluation = -np.inf
        for orient, slot in self.action_space.legal_moves[self.currentPiece]:
            board_copy = deepcopy(self.board)
            top_copy = deepcopy(self.top)
            self.perform_action(board_copy, top_copy, orient, slot)
            max_evaluation = max(max_evaluation, self.evaluate_board(board_copy, top_copy))
        return max_evaluation

    def reset(self):
        self.board = [[0] * Col for i in range(Row)]
        self.top = [0] * Col
        self.evaluation = 0
        self.total_score = 0
        self.currentPiece = self.new_piece()
        self.nextPiece = self.new_piece()
        observation = (deepcopy(self.board), self.currentPiece, self.nextPiece)
        return observation

    def render(self, mode='human', close=False):
        pass

    def close(self):
        pass

    def seed(self, seed=None):
        if seed is not None:
            self.randomness.seed = seed
        return self.randomness.seed

    def configure(self, *args, **kwargs):
        pass

    def new_piece(self):
        return self.randomness.randrange(0, Num_Types)

    def perform_action(self, board, top, orient, slot):
        reward = 0
        is_done = False
        height = top[slot] - pBottom[self.currentPiece][orient][0]
        for c in range(pWidth[self.currentPiece][orient]):
            height = max(height, top[slot + c] - pBottom[self.currentPiece][orient][c])

        if height + pHeight[self.currentPiece][orient] >= Row:
            is_done = True
            #death penalty
            return -1000, is_done

        for i in range(pWidth[self.currentPiece][orient]):
            for h in range(height + pBottom[self.currentPiece][orient][i], height + pTop[self.currentPiece][orient][i]):
                board[h][i + slot] = 1

        for c in range(pWidth[self.currentPiece][orient]):
            top[slot + c] = height + pTop[self.currentPiece][orient][c]

        for r in range(height + pHeight[self.currentPiece][orient] - 1, height - 1, -1):
            full = True
            for c in range(Col):
                if board[r][c] == 0:
                    full = False
                    break

            if full:
                reward = reward + 1
                for c in range(Col):
                    for i in range(r, top[c]):
                        board[i][c] = board[i + 1][c]
                    top[c] = top[c] - 1
                    while top[c] >= 1 and board[top[c] - 1][c] == 0:
                        top[c] = top[c] - 1

        return reward, is_done

    def evaluate_board(self, board, top):
        total_height = sum(top)
        diff_height = 0
        for i in range(Col - 1):
            diff_height = diff_height + abs(top[i] - top[i+1])
        holes = 0
        for i in range(Row):
            for j in range(Col):
                if board[i][j] == 0 and i < top[j]:
                    holes = holes + 1
        # arbitrary reward function from online source
        return -0.510066 * total_height - 0.184483 * diff_height - 0.35663 * holes

    class ActionSpace(Space):

        env = None
        random_action = Random()
        random_action.seed = 0
        legal_moves= []
        n = 0

        def __init__(self, env):
            self.env = env
            self.initialise_legal_moves()

        def contains(self, x):
            pass

        def sample(self, seed=None):
            if seed is not None:
                self.random_action.seed = seed
            piece_type = self.env.currentPiece
            choice_action = self.random_action.randrange(len(self.legal_moves[piece_type]))
            return choice_action

        def initialise_legal_moves(self):
            for i in range(Num_Types):
                n = 0
                for j in range(pOrients[i]):
                    n = n + Col+ 1 - pWidth[i][j]
                self.n = max(self.n, n)
                type_i_actions = []
                for j in range(pOrients[i]) :
                    for k in range(Col + 1 - pWidth[i][j]) :
                        action = (j, k)
                        type_i_actions.append(action)
                self.legal_moves.append(type_i_actions)