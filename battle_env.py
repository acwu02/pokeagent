import random
import gym

import numpy as np
import copy

from gym import spaces

from move import StruggleMove

STRUGGLE_MOVE = StruggleMove()

TYPE2IDX = {
    "normal":   0,
    "fire":     1,
    "water":    2,
    "electric": 3,
    "grass":    4,
    "ice":      5,
    "fighting": 6,
    "poison":   7,
    "ground":   8,
    "flying":   9,
    "psychic":  10,
    "bug":      11,
    "rock":     12,
    "ghost":    13,
    "dragon":   14,
    "dark":     15,
    "steel":    16,
    "fairy":    17,
}

IDX2TYPE = {v: k for k, v in TYPE2IDX.items()}

STATE_VEC_DIMS = 37

class PokemonBattleEnv(gym.Env):
    """
    A single-battle Pokémon environment (singles, no items, no weather).
    The learning agent always plays as trainer1; trainer2 can be scripted,
    random, or self-play.
    """
    metadata = {"render_modes": ["ansi"]}

    def __init__(self, trainer1, trainer2, max_turns=100, print_flag=False):
        super().__init__()

        self.t1 = trainer1
        self.t2 = trainer2

        self.turns = 0
        self.max_turns = max_turns

        self.action_space = spaces.Discrete(10)

        high = np.array([1.0] * STATE_VEC_DIMS, dtype=np.float32)
        self.observation_space = spaces.Box(low=0.0, high=high, dtype=np.float32)

        self.force_switch = False
        self.print_flag = print_flag

    def reset(self, *, seed=None, options=None):
        self.turns = 0
        super().reset(seed=seed)

        self.t1 = self.t1.clone_fresh()
        self.t2 = self.t2.clone_fresh()

        self.t1.send_out_first_pokemon()
        self.t2.send_out_first_pokemon()

        if self.print_flag:

            print(f"{self.t1.name} sends out {self.t1.current_pokemon.name}")
            print(f"{self.t2.name} sends out {self.t2.current_pokemon.name}")

        return self._encode_state(), {}

    def step(self, action: int):
        """
        One half-turn from the learning agent, followed by the opponent’s reply.
        """
        done = False
        reward = 0.0

        if self.force_switch:
            _, slot = None, action - 4
            self.t1.switch_to(slot)

            if self.print_flag:
                print(f"{self.t1.name} sent out {self.t1.current_pokemon.name}")

            self.force_switch = False
            self.turns += 1
            return self._encode_state(), reward, done, False, {
                "legal_actions": self.legal_actions(self.t1)
            }

        if self.print_flag:
            print(f"\n=== Turn {self.turns} ===")

        t1_current_pokemon_copy = copy.deepcopy(self.t1.current_pokemon)
        t2_current_pokemon_copy = copy.deepcopy(self.t2.current_pokemon)

        opp_alive_before   = self.t2.count_usable_pokemon()
        agent_alive_before = self.t1.count_usable_pokemon()

        t1_move, t1_next_slot = self._execute_action(self.t1, action)

        if t1_next_slot is not None:
            self.t1.switch_to(t1_next_slot)

        opp_fainted = not self.t2.has_usable_pokemon()
        agent_fainted = not self.t1.has_usable_pokemon()


        t2_move, t2_next_slot = None, None
        if not (opp_fainted or agent_fainted):
            t2_move, t2_next_slot = self._opponent_turn(t1_current_pokemon_copy)


        if t1_next_slot is not None:
            self.t1.switch_to(t1_next_slot)

            reward -= 0.1

            if self.print_flag:
                print(f"{self.t1.name} withdraws {t1_current_pokemon_copy.name}")
                print(f"{self.t1.name} sends out {self.t1.current_pokemon.name}")
                self.t1.current_pokemon.just_switched = True


        if t2_next_slot is not None:
            self.t2.switch_to(t2_next_slot)

            if self.print_flag:
                print(f"{self.t2.name} withdraws {t2_current_pokemon_copy.name}")
                print(f"{self.t2.name} sends out {self.t2.current_pokemon.name}")
                self.t2.current_pokemon.just_switched = True

        attacker, attacker_trainer = self.get_turn_order(self.t1.current_pokemon, self.t2.current_pokemon)
        defender, defender_trainer = (self.t2.current_pokemon, self.t2) if attacker_trainer == self.t1 else (self.t1.current_pokemon, self.t1)

        attacker_trainer_move = t1_move if attacker_trainer == self.t1 else t2_move
        defender_trainer_move = t2_move if attacker_trainer == self.t1 else t1_move

        attacker_damage, defender_damage = 0.0, 0.0

        if attacker_trainer_move:
            defender_damage = attacker_trainer_move.calculate_damage(attacker, defender)
            defender.take_damage(defender_damage)

            if self.print_flag:
                print(f"{attacker.name} uses {attacker_trainer_move.name}")
                print(f"Deals {defender_damage:.2f} damage to {defender.name}")


        if defender.is_fainted():

            if self.print_flag:
                print(f"{defender.name} fainted.")

            if defender_trainer.has_usable_pokemon():
                if defender_trainer == self.t1:
                    self.force_switch = True
                else:
                    defender_trainer.send_out_next_pokemon(attacker)
                    if self.print_flag:
                        print(f"{defender_trainer.name} sends out {defender_trainer.current_pokemon.name}")

            else:

                if self.print_flag:
                    print(f"{defender_trainer.name} has no usable Pokémon left.")

                done = True

        else:

            if defender_trainer_move:
                attacker_damage = defender_trainer_move.calculate_damage(defender, attacker)
                attacker.take_damage(attacker_damage)

                if self.print_flag:
                    print(f"{defender.name} uses {defender_trainer_move.name}")
                    print(f"Deals {attacker_damage:.2f} damage to {attacker.name}")

                if attacker.is_fainted():

                    if self.print_flag:
                        print(f"{attacker.name} fainted.")

                    if attacker_trainer.has_usable_pokemon():
                        if attacker_trainer == self.t1:
                            self.force_switch = True
                        else:
                            attacker_trainer.send_out_next_pokemon(defender)
                            if self.print_flag:
                                print(f"{attacker_trainer.name} sends out {attacker_trainer.current_pokemon.name}")
                    else:

                        if self.print_flag:
                            print(f"{attacker_trainer.name} has no usable Pokémon left.")
                        done = True

        opp_fainted = not self.t2.has_usable_pokemon()
        agent_fainted = not self.t1.has_usable_pokemon()

        if opp_fainted or agent_fainted:
            done = True

        opp_alive_after   = self.t2.count_usable_pokemon()
        agent_alive_after = self.t1.count_usable_pokemon()

        if self.print_flag:
            print(f"Remaining of {self.t1.name}: {agent_alive_after}")
            print(f"Remaining of {self.t2.name}: {opp_alive_after}")

        if self.t1.current_pokemon.just_switched:
            self.t1.current_pokemon.just_switched = False
        if self.t2.current_pokemon.just_switched:
            self.t2.current_pokemon.just_switched = False

        agent_damage = attacker_damage if attacker_trainer == self.t1 else defender_damage
        opp_damage = defender_damage if attacker_trainer == self.t1 else attacker_damage

        reward += self._reward_delta(opp_alive_before, opp_alive_after,
                                agent_alive_before, agent_alive_after,
                                agent_damage, opp_damage)

        if self.print_flag:
            print(f"Reward for this turn: {reward:.2f}")

        self.turns += 1

        if self.turns >= self.max_turns:
            done = True
            info = {"timeout": True}
        else:
            info = {}

        if done:
            r_term = +25.0 if opp_fainted else -25.0
            reward += r_term

        return self._encode_state(), reward, done, False, {}

    def get_turn_order(self, p1, p2):
        if p1.speed >= p2.speed:
            return p1, self.t1
        else:
            return p2, self.t2

    def _execute_action(self, trainer, action_idx):

        legal = self.legal_actions(trainer)
        if action_idx not in legal:
            raise ValueError(f"Illegal action {action_idx} for {trainer.name}: "
                            f"legal set = {sorted(legal)}")

        mon = trainer.current_pokemon
        if action_idx <= 3:

            move = mon.moves[action_idx]
            if move.pp_remaining == 0:
              legal_moves = [i for i, mv in enumerate(mon.moves) if mv.pp_remaining > 0]

              if legal_moves:
                  action_idx = int(np.random.choice(legal_moves))
                  move = mon.moves[action_idx]

              else:
                  move = STRUGGLE_MOVE

            return (move, None)
        else:
            slot = action_idx - 4
            if slot < len(trainer.team) and not trainer.team[slot].is_fainted():
                return (None, slot)

    def _opponent_turn(self, agent_pokemon):

        matchup = self.t2.current_pokemon.get_total_matchup_multiplier(agent_pokemon)

        # If in a bad matchup, attempt to switch
        if matchup < 1.0 and random.random() < 1.0:

            best_pokemon_idx = self.t2.get_best_pokemon_idx(agent_pokemon)

            if best_pokemon_idx and self.t2.team[best_pokemon_idx] != self.t2.current_pokemon:

                return (None, best_pokemon_idx)
            else:
                move = self.t2.get_best_move(agent_pokemon)
                return (move, None)
        else:
            move = self.t2.get_best_move(agent_pokemon)
            return (move, None)


    def legal_actions(self, trainer):
        if self.force_switch:
            # can ONLY choose a healthy bench mon
            return [
                4 + j for j, mon in enumerate(trainer.team)
                if not mon.is_fainted() and mon is not trainer.current_pokemon
            ]

        # ── normal turn ──
        actions = []
        for i, mv in enumerate(trainer.current_pokemon.moves):
            if mv.pp_remaining > 0:
                actions.append(i)
        for j, mon in enumerate(trainer.team):
            if not mon.is_fainted() and mon is not trainer.current_pokemon:
                actions.append(4 + j)
        return actions

    def _reward_delta(self, opp_before, opp_after, agent_before, agent_after, agent_damage, opp_damage):
        r = 0.0                 # 0 per turn – remove the tax for now
        r +=  3.0 * (opp_before   - opp_after)   # +3 per KO
        r += -3.0 * (agent_before - agent_after) # -3 per death

        r += 0.01 * opp_damage           # +0.01 per damage dealt
        r -= 0.01 * agent_damage            # -0.01 per damage taken

        r -= 0.01
        if self.turns > self.max_turns // 2:
            r -= 0.1

        return r

    def _encode_state(self):
        # ***Minimal numeric encoding — improve for stronger play***
        def mon_vec(mon):
            hp_pct = mon.hp / mon.max_hp
            type1 = TYPE2IDX[mon.type1]
            type2 = TYPE2IDX.get(mon.type2, 0)
            move_pp = [mv.pp_remaining / mv.pp for mv in mon.moves]
            stats = [mon.attack/300, mon.defense/300,
                     mon.sp_attack/300, mon.sp_defense/300,
                     mon.speed/300]
            return [hp_pct, type1/18, type2/18] + move_pp + stats

        # return np.array(
        #     mon_vec(self.t1.current_pokemon)
        #     + mon_vec(self.t2.current_pokemon)
        #     + [self.turn / self.max_turns],
        #     dtype=np.float32
        # )
        bench_t1 = [m.is_fainted() for m in self.t1.team]
        bench_t2 = [m.is_fainted() for m in self.t2.team]
        return np.array(
            mon_vec(self.t1.current_pokemon) +
            mon_vec(self.t2.current_pokemon) +
            bench_t1 + bench_t2 +
            [self.turns / self.max_turns],
            dtype=np.float32
        )

    def outcome(self) -> dict:
        """Return a compact summary of the finished battle."""
        return {
            "winner":   self.t1.name if self.t1.has_usable_pokemon() else self.t2.name,
            "turns":    self.turns,                       # need to track this; see § 3
            "t1_remaining": sum(not p.is_fainted() for p in self.t1.team),
            "t2_remaining": sum(not p.is_fainted() for p in self.t2.team)
        }


