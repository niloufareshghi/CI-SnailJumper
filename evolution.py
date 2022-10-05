import copy
import random

import numpy as np
from player import Player


class Evolution:
    def __init__(self):
        self.game_mode = "Neuroevolution"
        # self.iter = 0
        # self.info = []

    def next_population_selection(self, players, num_players):
        """
        Gets list of previous and current players (μ + λ) and returns num_players number of players based on their
        fitness value.

        :param players: list of players in the previous generation
        :param num_players: number of players that we return
        """
        # TODO (Implement top-k algorithm here)

        # TODO (Additional: Implement roulette wheel here)
        total = sum([player.fitness for player in players])  # total fitness
        probs = [fit / total for fit in [player.fitness for player in players]]  # probability
        selected = list(np.random.choice(players, num_players, p=probs, replace=False))

        # TODO (Additional: Implement SUS here)

        # TODO (Additional: Learning curve)

        max_fit = max([player.fitness for player in players])
        print(max_fit)
        # avg_fit = total / len(players)
        # min_fit = min([player.fitness for player in players])
        # gen_info = [max_fit, avg_fit, min_fit]
        # self.info.append(gen_info)
        # self.iter += 1
        # if self.iter == 20:
        #     file = open("gen_info.txt", 'w')
        #     for i in self.info:
        #         for j in i:
        #             file.write("%s " % j)
        #         file.write("\n")
        #     file.close()

        return selected

    def mutate(self, child: Player):

        for b in child.nn.biases:
            b += np.random.normal(0, 0.1, b.shape)

        for w in child.nn.weights:
            w += np.random.normal(0, 0.1, w.shape)

        return child

    def crossover(self, parent1, parent2, child: Player):
        w1 = parent1.nn.weights
        w2 = parent2.nn.weights
        b1 = parent1.nn.biases
        b2 = parent2.nn.biases

        for i in range(2):
            b_sieve = np.random.randint(2, size=b1[i].shape)
            child.nn.biases.append(b1[i] * b_sieve + b2[i] * (b_sieve ^ 1))
        for i in range(2):
            w_sieve = np.random.randint(2, size=w1[i].shape)
            child.nn.weights.append(w1[i] * w_sieve + w2[i] * (w_sieve ^ 1))

        return child

    def generate_new_population(self, num_players, prev_players=None):
        """
        Gets survivors and returns a list containing num_players number of children.

        :param num_players: Length of returning list
        :param prev_players: List of survivors
        :return: A list of children
        """
        first_generation = prev_players is None
        if first_generation:
            return [Player(self.game_mode) for _ in range(num_players)]
        else:
            # TODO ( Parent selection and child generation )
            total = sum([player.fitness for player in prev_players])  # total fitness
            probs = [fit / total for fit in [player.fitness for player in prev_players]]  # probability
            parents = np.random.choice(prev_players, num_players, p=probs)

            children = [self.clone_player(p) for p in parents]

            for i in range(num_players):
                p1, p2 = np.random.choice(parents, 2, p=probs)
                child = self.crossover(p1, p2, children[i])
                children.append(child)

            children = [self.mutate(c) if random.random() < 0.7 else c for c in children]

            return children

    def clone_player(self, player):
        """
        Gets a player as an input and produces a clone of that player.
        """
        new_player = Player(self.game_mode)
        new_player.nn = copy.deepcopy(player.nn)
        new_player.fitness = player.fitness
        return new_player
