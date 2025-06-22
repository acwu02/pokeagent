import math
import requests

class Pokemon:

    _SPECIES_CACHE = {}
    _TYPE_CACHE    = {}

    def __init__(self, name, moves, level=100, use_cache=True):
        self.name  = name.lower().replace(" ", "-")
        self.level = level

        self.moves = [mv.clone_fresh() for mv in moves]

        self._load_species_data(use_cache)

        self.just_switched = False
        self._init_battle_state()

    def clone_fresh(self):

        for move in self.moves:
            move = move.clone_fresh()
        return Pokemon(self.name, self.moves, self.level, use_cache=True)

    def take_damage(self, damage: int):

        self.damage_taken_last = max(0, damage)
        self.hp = max(0, self.hp - self.damage_taken_last)

    def is_fainted(self) -> bool:
        return self.hp <= 0

    def heal(self):
        self.hp = self.max_hp
        self.damage_taken_last = 0

    def get_matchup_multiplier(self, attacking_type: str) -> float:

        m = 1.0
        for t in (self.type1, self.type2) if self.type2 else (self.type1,):
            chart = Pokemon._TYPE_CACHE[t]
            if attacking_type in chart["double_damage_from"]:
                m *= 2.0
            elif attacking_type in chart["half_damage_from"]:
                m *= 0.5
            elif attacking_type in chart["no_damage_from"]:
                m *= 0.0
        return m

    def _init_battle_state(self):
        bp = self.base_stats
        L  = self.level

        self.max_hp   = math.floor(((2 * bp["hp"] * L) / 100) + L + 10)
        self.attack   = math.floor(((2 * bp["attack"]   * L) / 100) + 5)
        self.defense  = math.floor(((2 * bp["defense"]  * L) / 100) + 5)
        self.sp_attack   = math.floor(((2 * bp["sp_attack"]  * L) / 100) + 5)
        self.sp_defense   = math.floor(((2 * bp["sp_defense"] * L) / 100) + 5)
        self.speed    = math.floor(((2 * bp["speed"]    * L) / 100) + 5)

        self.hp                = self.max_hp
        self.damage_taken_last = 0

    def _load_species_data(self, use_cache: bool):

        if use_cache and self.name in Pokemon._SPECIES_CACHE:
            meta = Pokemon._SPECIES_CACHE[self.name]
        else:
            url = f"https://pokeapi.co/api/v2/pokemon/{self.name}"
            r   = requests.get(url, timeout=15)
            if r.status_code != 200:
                raise RuntimeError(f"Failed to fetch Pokémon '{self.name}'")
            data = r.json()

            types = [t["type"]["name"] for t in data["types"]]
            stats = { stat["stat"]["name"]: stat["base_stat"]
                      for stat in data["stats"] }

            meta = {
                "type1": types[0],
                "type2": types[1] if len(types) > 1 else None,
                "base_stats": {
                    "hp":            stats["hp"],
                    "attack":        stats["attack"],
                    "defense":       stats["defense"],
                    "sp_attack":     stats["special-attack"],
                    "sp_defense":    stats["special-defense"],
                    "speed":         stats["speed"],
                }
            }
            Pokemon._SPECIES_CACHE[self.name] = meta

        self.type1     = meta["type1"]
        self.type2     = meta["type2"]
        self.base_stats = meta["base_stats"]

        for t in (self.type1, self.type2) if self.type2 else (self.type1,):
            if t not in Pokemon._TYPE_CACHE:
                self._fetch_type_chart(t)

    def get_move_matchup_multiplier(self, other):

        move_multipliers = {}

        for move in self.moves:
            if move.is_status:
                continue
            attacker_type = move.type
            multiplier = other.get_matchup_multiplier(attacker_type)
            move_multipliers[move.name] = multiplier

        return move_multipliers

    def get_total_matchup_multiplier(self, other):

        type_mult_1 = other.get_matchup_multiplier(self.type1)
        type_mult_2 = other.get_matchup_multiplier(self.type2) if self.type2 else 1.0
        type_matchup = type_mult_1 * type_mult_2

        move_effectiveness = [
            other.get_matchup_multiplier(move.type)
            for move in self.moves if not move.is_status and hasattr(move, "type")
        ]
        if move_effectiveness:
            best_move_matchup = max(move_effectiveness)
        else:
            best_move_matchup = 0.5

        return 0.4 * type_matchup + 0.6 * best_move_matchup


    @staticmethod
    def _fetch_type_chart(t: str):
        url = f"https://pokeapi.co/api/v2/type/{t}"
        r   = requests.get(url, timeout=15)
        if r.status_code != 200:
            raise RuntimeError(f"Failed to fetch type chart for '{t}'")
        rel = r.json()["damage_relations"]
        Pokemon._TYPE_CACHE[t] = {
            "double_damage_from": [x["name"] for x in rel["double_damage_from"]],
            "half_damage_from":   [x["name"] for x in rel["half_damage_from"]],
            "no_damage_from":     [x["name"] for x in rel["no_damage_from"]],
        }

    def __str__(self):
        return (f"{self.name.title()} (Lv{self.level}) "
                f"[{self.type1}/{self.type2 or '--'}] "
                f"HP {self.hp}/{self.max_hp}")