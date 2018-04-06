from src.Tetris_Env import TetrisEnv

env = TetrisEnv()
action = env.action_space
_,reward, _,_ = env.step(action.sample())
env.step(action.sample())
print(env.top)
for i in range(20,-1,-1):
    print(env.board[i])
print(reward)

