import random

class Trainer:
    def __init__(self, name, pokemon_team):
        self.name = name
        self.team = pokemon_team
        self.current_pokemon = pokemon_team[0]

    def get_active_pokemon(self):
        return [p for p in self.team if not p.is_fainted()]

    def count_usable_pokemon(self):
        return len(self.get_active_pokemon())

    def has_usable_pokemon(self):
        return any(not p.is_fainted() for p in self.team)

    def choose_action(self, opponent_pokemon):
        if not self.current_pokemon or not opponent_pokemon:
            return

        matchup = self.current_pokemon.get_total_matchup_multiplier(opponent_pokemon)

        if matchup < 1.0 and random.random() < 0.7:
            next_pokemon = self.switch_pokemon(opponent_pokemon, exclude=self.current_pokemon)

            if next_pokemon and next_pokemon != self.current_pokemon:
                self.current_pokemon = next_pokemon
            else:
                return self.current_pokemon.choose_move(opponent_pokemon)
        else:
            return self.current_pokemon.choose_move(opponent_pokemon)


    def switch_to(self, slot):
        self.current_pokemon = self.team[slot]

    def get_best_pokemon_idx(self, opponent):

        def get_score(slot):
            return self.team[slot].get_total_matchup_multiplier(opponent)

        alive_indices = [
            i for i, mon in enumerate(self.team)
            if not mon.is_fainted() and mon is not self.current_pokemon
        ]

        if not alive_indices:
            return None
        return max(alive_indices, key=get_score)

    def get_best_move(self, opponent):
        damage_moves = {
            move: move.calculate_damage(self.current_pokemon, opponent) for move in self.current_pokemon.moves if not move.is_status
        }

        move = max(damage_moves, key=damage_moves.get)
        return move

    def send_out_next_pokemon(self, opponent):
        next_pokemon_idx = self.get_best_pokemon_idx(opponent)
        next_pokemon = self.team[next_pokemon_idx] if next_pokemon_idx is not None else None

        if next_pokemon and next_pokemon != self.current_pokemon:
            self.switch_to(next_pokemon_idx)
        else:
            return None

    def send_out_first_pokemon(self):
        self.current_pokemon = self.team[0]

        return self.current_pokemon

    def clone_fresh(self):
        return Trainer(
            self.name,
            [mon.clone_fresh() for mon in self.team]
        )

class TrainerAgent(Trainer):
    def __init__(self, name, team, policy="scripted"):
        self.name  = name
        self.team  = team
        self.policy = policy
        self.current_pokemon = self.team[0]

    def choose_action(self, opponent):
        if self.policy == "learner":
            return None
        elif self.policy == "random":
            legal   = [m for m in self.current_pokemon.moves if m.pp_remaining > 0]
            return random.choice(legal)
        else:
            best = max(
                self.current_pokemon.moves,
                key=lambda mv: mv.theoretical_damage(self.current_pokemon, opponent)
            )
            return best

    def clone_fresh(self):
        return TrainerAgent(
            self.name,
            [mon.clone_fresh() for mon in self.team],
            self.policy
        )