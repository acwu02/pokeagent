# PokéAgent

This is a deep reinforcement learning agent designed to find optimal strategies for winning Pokémon battles. It implements Double DQN to prevent overestimation of learned Q-values. [1] 
It runs on [PokéAPI](https://pokeapi.co/), and a basic Pokémon battle environment developed by me. Currently, only damaging moves are supported.

## Setup

Install requirements:

```
pip install -r requirements.txt
```

`example_trainers.py` contains example code on how to instantiate Pokémon trainers. `train.py` and `eval.py` contain example code on how to train and evaluate agents, respectively. 

## References

[1] Van Hasselt, Hado, Arthur Guez, and David Silver. "Deep reinforcement learning with double q-learning." Proceedings of the AAAI conference on artificial intelligence. Vol. 30. No. 1. 2016.
