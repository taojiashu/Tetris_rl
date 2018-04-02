from rl.core import Env, Space
from copy import deepcopy
from src.configuration import pOrients, pWidth, pHeight, pTop, pBottom
from random import Random

class TetrisEnv(Env):
    Num_Types = 7
    Col = 10
    Row = 21
    randomness = Random()
    randomness.seed = 0

    board = None
    top = None
    currentPiece = None
    nextPiece = None
    info = None

    def __init__(self):
        self.board = [[0] * self.Col for i in range(self.Row)]
        self.top = [0] * self.Col
        self.currentPiece = self.new_piece()
        self.nextPiece = self.new_piece()
        self.action_space = self.ActionSpace(self)

    def step(self, action):
        orient, slot = action
        reward, is_done = self.perform_action(orient, slot)
        self.currentPiece = self.nextPiece
        self.nextPiece = self.new_piece()
        observation = (deepcopy(self.board), self.currentPiece, self.nextPiece)
        return observation, reward, is_done, self.info

    def reset(self):
        self.board = [[0]*self.Col for i in range(self.Row)]
        self.currentPiece = self.new_piece()
        self.nextPiece = self.new_piece()

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
        return self.randomness.randrange(0,self.Num_Types)

    def perform_action(self, orient, slot):
        reward = 0.1
        is_done = False
        height = self.top[slot] - pBottom[self.currentPiece][orient][0]
        for c in range(pWidth[self.currentPiece][orient]):
            height = max(height,self.top[slot+c]-pBottom[self.currentPiece][orient][c])

        if height+pHeight[self.currentPiece][orient] >= self.Row:
            is_done = True

        for i in range(pWidth[self.currentPiece][orient]):
            for h in range(height+pBottom[self.currentPiece][orient][i], height+pTop[self.currentPiece][orient][i]):
                self.board[h][i+slot] = 1


        for c in range(pWidth[self.currentPiece][orient]):
            self.top[slot+c]=height+pTop[self.currentPiece][orient][c]


        for r in range(height+pHeight[self.currentPiece][orient]-1, height-1,-1):
            full = True
            for c in range(self.Col):
                if self.board[r][c] == 0:
                    full = False
                    break

            if full:
                reward = reward + 1
                for c in range(self.Col):
                    for i in range(r, self.top[c]):
                        self.board[i][c] = self.board[i+1][c]
                    self.top[c] = self.top[c] - 1
                    while self.top[c]>=1 and self.board[self.top[c] - 1][c]==0:
                        self.top[c]= self.top[c]-1

        return reward, is_done

    class ActionSpace(Space):

        env = None
        random_action = Random()
        random_action.seed = 0

        def __init__(self, env):
            self.env = env

        def contains(self, x):
            pass

        def sample(self, seed=None):
            if seed is not None:
                self.random_action.seed = seed
