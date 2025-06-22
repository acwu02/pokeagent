from pokemon import Pokemon
from move import Move
from pokemon_trainer import Trainer, TrainerAgent

class Dawn(TrainerAgent):
    def __init__(self, name="Dawn"):
        self.team = [
            Pokemon("infernape", [Move("flamethrower"), Move("close-combat"), Move("thunder-punch"), Move("earthquake")]),
            Pokemon("staraptor", [Move("brave-bird"), Move("close-combat"), Move("roost"), Move("steel-wing")]),
            Pokemon("electivire", [Move("thunderbolt"), Move("cross-chop"), Move("ice-punch"), Move("fire-punch")]),
            Pokemon("mamoswine", [Move("earthquake"), Move("icicle-crash"), Move("stone-edge"), Move("ancient-power")]),
            Pokemon("bronzong", [Move("extrasensory"), Move("earthquake"), Move("stealth-rock"), Move("flash-cannon")]),
            Pokemon("tentacruel", [Move("scald"), Move("sludge-bomb"), Move("toxic-spikes"), Move("ice-beam")])
        ]
        super().__init__(name, self.team)

class Cynthia(Trainer):
    def __init__(self, name="Cynthia"):
        self.team = [
            Pokemon("spiritomb", [Move("shadow-ball"), Move("dark-pulse"), Move("will-o-wisp"), Move("silver-wind")]),
            Pokemon("garchomp", [Move("earthquake"), Move("dragon-claw"), Move("stone-edge"), Move("flamethrower")]),
            Pokemon("lucario", [Move("close-combat"), Move("aura-sphere"), Move("extreme-speed"), Move("ice-punch")]),
            Pokemon("milotic", [Move("surf"), Move("ice-beam"), Move("recover"), Move("bulldoze")]),
            Pokemon("roserade", [Move("energy-ball"), Move("sludge-bomb"), Move("toxic-spikes"), Move("extrasensory")]),
            Pokemon("togekiss", [Move("air-slash"), Move("thunder-wave"), Move("roost"), Move("aura-sphere")])
        ]
        super().__init__(name, self.team)