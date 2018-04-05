from rl.core import Env, Space
from copy import deepcopy
from src.configuration import pOrients, pWidth, pHeight, pTop, pBottom, Num_Types, Col, Row
from random import Random


class TetrisEnv(Env):
    randomness = Random()
    randomness.seed = 0

    board = None
    top = None
    currentPiece = None
    nextPiece = None
    info = {"dummp":1}
    evaluation = 0

    def __init__(self):
        self.board = [[0] * Col for i in range(Row)]
        self.top = [0] * Col
        self.currentPiece = self.new_piece()
        self.nextPiece = self.new_piece()
        self.action_space = self.ActionSpace(self)

    def step(self, action):
        if action < len(self.action_space.legal_moves[self.currentPiece]):
            orient, slot = self.action_space.legal_moves[self.currentPiece][action]
        else:
            orient, slot = self.action_space.legal_moves[self.currentPiece][self.action_space.sample()]
        true_reward, is_done = self.perform_action(orient, slot)
        new_evaluation = self.evaluate_board()
        reward = new_evaluation - self.evaluation + 100
        self.evaluation = new_evaluation
        self.currentPiece = self.nextPiece
        self.nextPiece = self.new_piece()
        observation = (deepcopy(self.board), self.currentPiece, self.nextPiece)
        return observation, reward, is_done, self.info

    def reset(self):
        self.board = [[0] * Col for i in range(Row)]
        self.top = [0] * Col
        evaluation = 0
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

    def perform_action(self, orient, slot):
        reward = 0
        is_done = False
        height = self.top[slot] - pBottom[self.currentPiece][orient][0]
        for c in range(pWidth[self.currentPiece][orient]):
            height = max(height, self.top[slot + c] - pBottom[self.currentPiece][orient][c])

        if height + pHeight[self.currentPiece][orient] >= Row:
            is_done = True
            return reward, is_done

        for i in range(pWidth[self.currentPiece][orient]):
            for h in range(height + pBottom[self.currentPiece][orient][i], height + pTop[self.currentPiece][orient][i]):
                self.board[h][i + slot] = 1

        for c in range(pWidth[self.currentPiece][orient]):
            self.top[slot + c] = height + pTop[self.currentPiece][orient][c]

        for r in range(height + pHeight[self.currentPiece][orient] - 1, height - 1, -1):
            full = True
            for c in range(Col):
                if self.board[r][c] == 0:
                    full = False
                    break

            if full:
                reward = reward + 1
                for c in range(Col):
                    for i in range(r, self.top[c]):
                        self.board[i][c] = self.board[i + 1][c]
                    self.top[c] = self.top[c] - 1
                    while self.top[c] >= 1 and self.board[self.top[c] - 1][c] == 0:
                        self.top[c] = self.top[c] - 1

        return reward, is_done

    def evaluate_board(self):
        total_height = sum(self.top)
        max_height = max(self.top)
        diff_height = 0
        for i in range(Col - 1):
            diff_height = diff_height + abs(self.top[i] - self.top[i+1])
        holes = 0
        for i in range(Row):
            for j in range(Col):
                if self.board[i][j] == 0 and i < self.top[j]:
                    holes = holes + 1
        # arbitrary reward function
        return - 0.1 * total_height - max_height - 0.1 * diff_height - holes

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