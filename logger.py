import csv
import gym

class BattleLogger(gym.Wrapper):
    def __init__(self, env, path="battle_log.csv", print_flag=False):
        super().__init__(env)
        self.path = path
        self._file = open(self.path, "a", newline="")
        self._writer = None
        self.print_flag = print_flag

    def reset(self, **kwargs):
        self._ep_reward = 0.0
        return super().reset(**kwargs)

    def step(self, action):
        obs, reward, done, trunc, info = self.env.step(action)
        self._ep_reward += reward

        if done or trunc:

            if self.print_flag:
                print(f"Episode reward: {self._ep_reward:.2f}")

            out = self.env.outcome()
            row = {
                "episode": info.get("episode"),
                **out,
                "reward": self._ep_reward
            }
            if self._writer is None:
                self._writer = csv.DictWriter(self._file, fieldnames=row.keys())
                if self._file.tell() == 0:
                    self._writer.writeheader()
            self._writer.writerow(row)
            self._file.flush()
        return obs, reward, done, trunc, info
