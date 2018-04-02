from rl.core import Env
from copy import deepcopy
from configuration import pOrients, pWidth, pHeight, pTop, pBottom,
import random

class TetrisEnv(Env):
    Num_Types = 7
    Col = 10
    Row = 21
    random_seed = 0;
    board = [[0]*Col for i in range(Row)]
    top = [0]*Col
    currentPiece = None
    nextPiece = None
    info = None

    def step(self, action):
        reward, isDone = self.perform_action(self.board, action)
        self.currentPiece = self.nextPiece
        self.nextPiece = self.new_piece()
        observation = (deepcopy(self.board), self.currentPiece, self.nextPiece)
        return observation, reward, isDone, self.info

    def reset(self):
        self.random_seed = 0

    def render(self, mode='human', close=False):
        pass

    def close(self):
        pass

    def seed(self, seed=None):
        random_seed = seed
        return random_seed

    def configure(self, *args, **kwargs):
        pass

    def new_piece(self):
        return random.randrange(0,self.Num_Types)

    def perform_action(self, action):
        reward = 0.1
        isDone = False
        orient, slot = action
        height = self.top[slot] - pBottom[self.currentPiece][orient][0]
        for c in range(pWidth[self.currentPiece][orient]):
            height = max(height,self.top[slot+c]-pBottom[self.currentPiece][orient][c])

        if (height+pHeight[self.currentPiece][orient] >= self.Row):
            isDone = True

        for i in range(pWidth[self.currentPiece][orient]):
			for h in range(height+pBottom[self.currentPiece][orient][i], height+pTop[self.currentPiece][orient][i]):
				self.board[h][i+slot] = 1


		for c in range(pWidth[self.currentPiece][orient]):
            self.top[slot+c]=height+pTop[self.currentPiece][orient][c]


        for r in range(height+pHeight[self.currentPiece][orient]-1, height-1,-1):
            full = True
            for c in range(self.Col):
                if(self.board[r][c] == 0):
                    full = False
                    break


            if (full) :
                reward = reward + 1
                for c in range(self.Col):
                    for i in range(r, self.top[c]):
                        self.board[i][c] = self.board[i+1][c]
                    self.top[c] = self.top[c] - 1
                    while (self.top[c]>=1 and self.board[self.top[c]-1][c]==0):
                        self.top[c]= self.top[c]-1

        return reward, isDone