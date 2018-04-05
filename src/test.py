from src.Tetris_Env import TetrisEnv

env = TetrisEnv()
action = env.action_space
_,reward, _,_ = env.step(action.sample())
print(env.top)
print(reward)

