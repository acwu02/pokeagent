import random
import requests

class Move:
    _CACHE = {}

    def __init__(self, name, *, use_cache=True):
        self.name = name.lower().replace(" ", "-")
        self._load_data(use_cache)

        self.pp_remaining = self.pp
        self.is_used      = False

    def apply(self, attacker, defender):
        if self.pp_remaining == 0:
            raise RuntimeError(f"{self.name} is out of PP!")
        self.pp_remaining -= 1
        self.is_used = True

        dmg = self.calculate_damage(attacker, defender)
        defender.take_damage(dmg)
        defender.damage_taken_last = dmg
        attacker.damage_taken_last = 0
        return dmg

    def theoretical_damage(self, attacker, defender):
        return self._base_damage(attacker, defender) * \
               attacker.get_matchup_multiplier(self.type)

    def clone_fresh(self):
        return Move(self.name, use_cache=True)

    def __str__(self):
        return (f"{self.name} (Type {self.type}, Pow {self.power}, "
                f"Acc {self.accuracy}, PP {self.pp_remaining}/{self.pp})")

    def calculate_damage(self, attacker, defender):
        effectiveness = defender.get_matchup_multiplier(self.type)

        if self.type in (attacker.type1, attacker.type2):
            effectiveness *= 1.5

        rand = random.uniform(0.85, 1.0)

        return int(self._base_damage(attacker, defender) *
                   effectiveness * rand)

    def _base_damage(self, attacker, defender):
        atk_stat = attacker.sp_attack if self.is_special else attacker.attack
        def_stat = defender.sp_defense if self.is_special else defender.defense
        lvl      = attacker.level
        power    = self.power or 0

        return (((2 * lvl / 5 + 2) * power * (atk_stat / def_stat)) / 50) + 2

    def _load_data(self, use_cache):

        if use_cache and self.name in Move._CACHE:
            meta = Move._CACHE[self.name]
        else:
            meta = self._fetch_from_api(self.name)
            Move._CACHE[self.name] = meta

        self.type        = meta["type"]
        self.power       = meta["power"]
        self.accuracy    = meta["accuracy"]
        self.pp          = meta["pp"]
        self.is_status   = meta["is_status"]
        self.is_special  = meta["is_special"]
        self.is_physical = meta["is_physical"]

    @staticmethod
    def _fetch_from_api(name=None):
        """One real HTTP call; cached thereafter."""
        if name is None:
            raise ValueError("Move name required")
        url = f"https://pokeapi.co/api/v2/move/{name}"
        r   = requests.get(url, timeout=15)
        if r.status_code != 200:
            raise RuntimeError(f"Failed to fetch data for move '{name}'")

        data = r.json()
        dmg_class = data["damage_class"]["name"]

        return dict(
            type        = data["type"]["name"],
            power       = data["power"]     or 0,
            accuracy    = data["accuracy"]  or 100,
            pp          = data["pp"],
            is_status   = dmg_class == "status",
            is_special  = dmg_class == "special",
            is_physical = dmg_class == "physical",
        )

class StruggleMove(Move):
    def __init__(self):
        super().__init__(
            name="struggle"
        )

    def _deduct_pp(self):
        pass