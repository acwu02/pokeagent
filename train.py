import random
from collections import deque, namedtuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from battle_env import PokemonBattleEnv
from logger import BattleLogger
from qnet import QNet

from example_trainers import Dawn, Cynthia

BATCH = 64
GAMMA = 0.99
TARGET_SYNC = 1_000
UPDATE_START = 1_000

NUM_STEPS = 20_000

dawn = Dawn()
cynthia = Cynthia()

env = PokemonBattleEnv(dawn, cynthia)
env = BattleLogger(env)

obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.n
policy  = QNet(obs_dim, act_dim)
target  = QNet(obs_dim, act_dim)
target.load_state_dict(policy.state_dict())
opt     = optim.Adam(policy.parameters(), lr=1e-3)

memory  = deque(maxlen=100_000)
Transition = namedtuple("T", "s a r s2 d legal_next")

EPS_START, EPS_END, EPS_DECAY = 1.0, 0.05, 150_000
steps_done = 0

INF = 1e9

def select_action(policy, state, legal):

    global steps_done

    # ε-schedule
    eps = EPS_END + (EPS_START - EPS_END) * \
          np.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1

    if random.random() < eps:
        return int(random.choice(legal))

    with torch.no_grad():
        s = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        q = policy(s).squeeze(0)

        q_masked = q.clone()
        q_masked[:] = q_masked - INF
        q_masked[legal] = q[legal]

        if torch.isinf(q_masked.max()):
            return int(random.choice(legal))

        return int(q_masked.argmax())

def mask_q_values(q_vals, legal_batch):

    q_masked = q_vals.clone()
    q_masked[:] = q_masked - INF
    for row, legal in enumerate(legal_batch):
        if legal:
            q_masked[row, legal] = q_vals[row, legal]
    return q_masked

for episode in tqdm(range(NUM_STEPS)):
    state, _ = env.reset()
    done = False
    info = {"episode": episode}
    while not done:

        legal = env.unwrapped.legal_actions(env.t1)
        a = select_action(policy, state, legal)

        s2, r, done, _, _ = env.step(a)

        legal_next = env.unwrapped.legal_actions(env.t1) if not done else []
        memory.append(Transition(state, a, r, s2, done, legal_next))
        state = s2

        if len(memory) >= UPDATE_START:
            batch = random.sample(memory, BATCH)
            s, a, r, s2, d, legal_next = zip(*batch)

            s   = torch.as_tensor(np.stack(s),  dtype=torch.float32)
            a   = torch.as_tensor(a,          dtype=torch.int64).unsqueeze(1)
            r   = torch.as_tensor(r,          dtype=torch.float32).unsqueeze(1)
            s2  = torch.as_tensor(np.stack(s2), dtype=torch.float32)
            d   = torch.as_tensor(d,          dtype=torch.float32).unsqueeze(1)

            q    = policy(s).gather(1, a)

            q_next_online      = policy(s2)
            q_next_online_mask = mask_q_values(q_next_online,
                                              legal_next)

            a_star = q_next_online_mask.argmax(1, keepdim=True)

            q_next_target = target(s2).gather(1, a_star)

            td   = r + GAMMA * q_next_target * (1 - d)
            loss = nn.functional.smooth_l1_loss(q, td.detach())

            opt.zero_grad()
            loss.backward()
            opt.step()

            if steps_done % TARGET_SYNC == 0:
                target.load_state_dict(policy.state_dict())

torch.save(policy.state_dict(), "policy.pt")