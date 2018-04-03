from src.Tetris_Env import TetrisEnv

class GeneticAgent(object):
    weight = None
    env = None
    action_space = None

    def __init__(self, weight):
        self.weight = weight
        env = TetrisEnv()
        action_space = env.action_space

    def play(self):
        observaton, _, is_done, _ = self.env.reset()
        while not is_done:
            action = self.choose_action(observaton)
            observation, reward, is_done, _ = self.env.step(action)
        return reward

    def choose_action(self, obseration):

        return action
