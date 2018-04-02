from src.Tetris_Env import TetrisEnv


env = TetrisEnv()
action = env.action_space.sample()
(env.step(action))
for i in range(20, -1, -1):
    print (env.board[i])
