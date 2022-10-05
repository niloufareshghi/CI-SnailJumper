import numpy as np
import math
import pygame
from variables import global_variables
from nn import NeuralNetwork


class Player(pygame.sprite.Sprite):
    def __init__(self, game_mode):
        super().__init__()

        # self.side = 'left'

        # loading images
        player_walk1 = pygame.image.load('Graphics/Player/player_walk_1.png').convert_alpha()
        player_walk2 = pygame.image.load('Graphics/Player/player_walk_2.png').convert_alpha()

        # rotating -90 degree and scaling by factor of 0.5
        player_walk1 = pygame.transform.rotozoom(player_walk1, -90, 0.5)
        player_walk2 = pygame.transform.rotozoom(player_walk2, -90, 0.5)

        # flipping vertically
        player_walk1 = pygame.transform.flip(player_walk1, flip_x=False, flip_y=True)
        player_walk2 = pygame.transform.flip(player_walk2, flip_x=False, flip_y=True)

        self.player_walk = [player_walk1, player_walk2]

        self.player_index = 0

        self.image = self.player_walk[self.player_index]
        self.rect = self.image.get_rect(midleft=(177, 656))

        self.player_gravity = 'left'
        self.gravity = 10
        self.game_mode = game_mode

        if self.game_mode == "Neuroevolution":
            self.fitness = 0  # Initial fitness

            layer_sizes = [6, 21, 1]  # TODO (Design your architecture here by changing the values)
            self.nn = NeuralNetwork(layer_sizes)

    def think(self, screen_width, screen_height, obstacles, player_x, player_y):
        """
        Creates input vector of the neural network and determines the gravity according to neural network's output.

        :param screen_width: Game's screen width which is 604.
        :param screen_height: Game's screen height which is 800.
        :param obstacles: List of obstacles that are above the player. Each entry is a dictionary having 'x' and 'y' of
        the obstacle as the key. The list is sorted based on the obstacle's 'y' point on the screen. Hence, obstacles[0]
        is the nearest obstacle to our player. It is also worthwhile noting that 'y' range is in [-100, 656], such that
        -100 means it is off screen (Topmost point) and 656 means in parallel to our player's 'y' point.
        :param player_x: 'x' position of the player
        :param player_y: 'y' position of the player
        """
        # TODO (change player's gravity here by calling self.change_gravity)

        diagonal = math.sqrt(screen_height ** 2 + 233 ** 2)

        min_dist_left = diagonal
        min_dist_right = diagonal
        min_dist_fly = diagonal

        obs_left = {'x': 177, 'y': -1e9}
        obs_right = {'x': 410, 'y': -1e9}
        obs_fly = {'x': -1e9, 'y': -1e9}

        def dist(px, py, qx, qy):
            d = math.sqrt((px - qx) ** 2 + (py - qy) ** 2)
            return d

        for obs in obstacles:
            if obs['x'] < 180:
                dist_left = dist(player_x, player_y, obs['x'], obs['y'])
                if dist_left < min_dist_left:
                    min_dist_left = dist_left
                    obs_left['y'] = obs['y']
                    obs_left['x'] = obs['x']
            if obs['x'] > 400:
                dist_right = dist(player_x, player_y, obs['x'], obs['y'])
                if dist_right < min_dist_right:
                    min_dist_right = dist_right
                    obs_right['y'] = obs['y']
                    obs_right['x'] = obs['x']
            if 177 < obs['x'] < 410:
                dist_fly = dist(player_x, player_y, obs['x'], obs['y'])
                if dist_fly < min_dist_fly:
                    min_dist_fly = dist_fly
                    obs_fly['x'] = obs['x']
                    obs_fly['y'] = obs['y']

        if obs_fly['y'] > -1e5 and obs_left['y'] > -1e5:
            dist_f_l = dist(obs_fly['x'], obs_fly['y'], obs_left['x'], obs_left['y'])
        else:
            dist_f_l = diagonal

        if obs_fly['y'] > -1e5 and obs_right['y'] > -1e5:
            dist_f_r = dist(obs_fly['x'], obs_fly['y'], obs_right['x'], obs_right['y'])
        else:
            dist_f_r = diagonal

        a = (player_x - 177) / 233

        # if a < 0.1:
        #     self.side = 'left'
        # elif a > 0.9:
        #     self.side = 'right'
        #
        # if self.side == 'right':
        #     temp = min_dist_left
        #     min_dist_left = min_dist_right
        #     min_dist_right = temp

        data = [min_dist_left / diagonal,
                min_dist_right / diagonal,
                min_dist_fly / diagonal,
                dist_f_l / diagonal,
                dist_f_r / diagonal,
                a
                ]

        output = self.nn.forward(np.reshape(np.array(data), (len(data), 1)))

        # if self.side == 'right':
        #     output[0][0] = 1 - output[0][0]

        if output[0][0] > 0.5:
            self.change_gravity('right')
        else:
            self.change_gravity('left')

    def change_gravity(self, new_gravity):
        """
        Changes the self.player_gravity based on the input parameter.
        :param new_gravity: Either "left" or "right"
        """
        new_gravity = new_gravity.lower()

        if new_gravity != self.player_gravity:
            self.player_gravity = new_gravity
            self.flip_player_horizontally()

    def player_input(self):
        """
        In manual mode: After pressing space from the keyboard toggles player's gravity.
        """
        if global_variables['events']:
            for pygame_event in global_variables['events']:
                if pygame_event.type == pygame.KEYDOWN:
                    if pygame_event.key == pygame.K_SPACE:
                        self.player_gravity = "left" if self.player_gravity == 'right' else 'right'
                        self.flip_player_horizontally()

    def apply_gravity(self):
        if self.player_gravity == 'left':
            self.rect.x -= self.gravity
            if self.rect.left <= 177:
                self.rect.left = 177
        else:
            self.rect.x += self.gravity
            if self.rect.right >= 430:
                self.rect.right = 430

    def animation_state(self):
        """
        Animates the player.
        After each execution, it increases player_index by 0.1. Therefore, after ten execution, it changes the
        player_index and player's frame correspondingly.
        """
        self.player_index += 0.1
        if self.player_index >= len(self.player_walk):
            self.player_index = 0

        self.image = self.player_walk[int(self.player_index)]

    def update(self):
        """
        Updates the player according to the game_mode. If it is "Manual", it listens to the keyboard. Otherwise the
        player changes its location based on `think` method.
        """
        if self.game_mode == "Manual":
            self.player_input()
        if self.game_mode == "Neuroevolution":
            obstacles = []
            for obstacle in global_variables['obstacle_groups']:
                if obstacle.rect.y <= 656:
                    obstacles.append({'x': obstacle.rect.x, 'y': obstacle.rect.y})

            self.think(global_variables['screen_width'],
                       global_variables['screen_height'],
                       obstacles, self.rect.x, self.rect.y)

        self.apply_gravity()
        self.animation_state()

    def flip_player_horizontally(self):
        """
        Flips horizontally to have a better graphic after each jump.
        """
        for i, player_surface in enumerate(self.player_walk):
            self.player_walk[i] = pygame.transform.flip(player_surface, flip_x=True, flip_y=False)
