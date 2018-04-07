from src.Tetris_Env import TetrisEnv
from src.tetris_generator import TetrisGenerator
generator = TetrisGenerator()


observation, action = generator.__getitem__(1)
print(observation)
print(action)
observation, action = generator.__getitem__(2)
print(observation)
print(action)
