import random
import torch
import numpy as np
from collections import defaultdict
from tqdm import tqdm

from pokeagent import Dawn, Cynthia, PokemonEnv, BattleLogger
from qnet import QNet

dawn = Dawn()
cynthia = Cynthia()

env = PokemonEnv(dawn, cynthia)
env = BattleLogger(env)

obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.n

policy  = QNet(obs_dim, act_dim)
policy.load_state_dict(torch.load("policy.pt"))

policy.eval()
for p in policy.parameters():
    p.requires_grad_(False)

dawn_eval     = Dawn()
cynthia_eval  = Cynthia()
eval_env      = BattleLogger(PokemonEnv(dawn_eval, cynthia_eval, print_flag=True))

state, _      = eval_env.reset()
done          = False
turn          = 0
total_reward  = 0.0

print("\n=== BEGIN EVALUATION BATTLE ===\n")

INF = 1e9
def greedy_action(state, legal):
    with torch.no_grad():
        s = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        q = policy(s).squeeze(0)

        q_masked = q.clone()
        q_masked[:]     = q_masked - INF
        q_masked[legal] = q[legal]

        if torch.isinf(q_masked.max()):
            return int(random.choice(legal))
        return int(q_masked.argmax())

def action_label(idx, env, trainer):
    if idx <= 3:
        move = trainer.current_pokemon.moves[idx].name
        return f"Move {idx}: {move}"
    else:
        slot   = idx - 4
        target = trainer.team[slot]
        return f"Switch → slot {slot} ({target.name})"

while not done:
    turn += 1
    t1  = eval_env.t1
    t2  = eval_env.t2
    legal = eval_env.unwrapped.legal_actions(t1)

    with torch.no_grad():
        s_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        q_all    = policy(s_tensor).squeeze(0)

    q_masked = q_all.clone()
    q_masked[:]       = q_masked - INF
    q_masked[legal]   = q_all[legal]

    print(f"\n[Turn {turn:>2}] {t1.current_pokemon.name} vs {t2.current_pokemon.name}")
    for idx in legal:
        label = action_label(idx, eval_env, t1)
        print(f"   {idx:2d} | {label:<25} Q = {q_masked[idx]:7.3f}")

    action = int(q_masked.argmax() if not torch.isinf(q_masked.max())
                 else random.choice(legal))

    print(f"   → Chosen action: {action} ({action_label(action, eval_env, t1)})")

    state, reward, done, _, _ = eval_env.step(action)
    total_reward += reward

print("\n=== END OF BATTLE ===")
winner = "Dawn (agent)" if eval_env.t1.has_usable_pokemon() else "Cynthia (opponent)"
print(f"Winner: {winner}")
print(f"Remaining of Dawn: {eval_env.t1.count_usable_pokemon()}")
print(f"Remaining of Cynthia: {eval_env.t2.count_usable_pokemon()}")
print(f"Total reward collected: {total_reward:.2f}")

def greedy_action(state, legal, policy, INF=1e9):
    with torch.no_grad():
        s = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        q = policy(s).squeeze(0)

        q_masked         = q.clone()
        q_masked[:]      = q_masked - INF
        q_masked[legal]  = q[legal]

        if torch.isinf(q_masked.max()):
            return int(random.choice(legal))
        return int(q_masked.argmax())

def run_battle(policy, verbose=False):

    dawn_eval     = Dawn()
    cynthia_eval  = Cynthia()
    env           = BattleLogger(PokemonEnv(dawn_eval, cynthia_eval,
                                           print_flag=verbose))

    state, _      = env.reset()
    done          = False
    total_reward  = 0.0
    turn          = 0
    INF           = 1e9

    while not done:
        turn += 1
        t1      = env.t1
        legal   = env.unwrapped.legal_actions(t1)
        action  = greedy_action(state, legal, policy, INF)
        state, r, done, _, _ = env.step(action)
        total_reward += r

    winner = "agent" if env.t1.has_usable_pokemon() else "opponent"
    return winner, total_reward, turn

def evaluate_many(policy, n_trials=200, seed=0, verbose_every=None):
    random.seed(seed)
    torch.manual_seed(seed)

    results = defaultdict(list)
    for i in tqdm(range(1, n_trials + 1)):
        verbose = (verbose_every is not None and i % verbose_every == 0)
        winner, reward, turns = run_battle(policy, verbose)
        results["winner"].append(winner)
        results["reward"].append(reward)
        results["turns"].append(turns)

        if verbose_every is None:
            print(f"[{i:>4}/{n_trials}]  winner={winner:8s}  "
                  f"R={reward:+6.2f}  T={turns}")

    wins      = results["winner"].count("agent")
    losses    = n_trials - wins
    avg_R     = np.mean(results["reward"])
    avg_turns = np.mean(results["turns"])

    print("\n=== Evaluation summary ===")
    print(f"Trials run        : {n_trials}")
    print(f"Agent wins/losses : {wins} / {losses} "
          f"({wins/n_trials:.1%} win-rate)")
    print(f"Avg total reward  : {avg_R:+.2f}")
    print(f"Avg battle length : {avg_turns:.1f} turns")

    return results

policy.eval()
for p in policy.parameters():
    p.requires_grad_(False)

seed = random.randint(0, 1000)
print(f"Random seed: {seed}")

_ = evaluate_many(policy, n_trials=1000, seed=seed, verbose_every=50)