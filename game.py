"""
**************************************

Created in 25.01.2022
by Aiyyskhan Alekseev

https://github.com/AiyyArt

https://opensea.io/collection/aiyyart-collection

ETH: 0x4e6c76f938d941e5b4bf1a11e3fd20f311e59df6

timirkhan@gmail.com

**************************************
"""

# *** path to PNG file ***
PDF_LOAD_PATH = "data/NeuralNetS_G0.gif" #C:/.../labirint_0/data/NeuralNetS_G0.gif" 

__author__ = "Aiyyskhan Alekseev"
__version__ = "0.2.0"


import math
import numpy as np
from PIL import Image
import pygame

from settings import *
import nn_1pl as nn
from player import Player
from map_file_lev0_2 import get_map
from drawing import Drawing
from ray_casting import RayCast

VAL = np.array([-1.0,-0.75,-0.5,-0.25,0.0,0.25,0.5,0.75,1.0])


def iter_frames(im):
    try:
        i = 0
        while 1:
            im.seek(i)
            yield im.copy().convert("RGBA")
            i += 1
    except EOFError:
        pass
    
def png2arr(frame, val):
    w_ids = (np.array(frame) / 255.0) * 8.0
        
    w_ids = np.rot90(w_ids, axes=(2,0))

    w_id_0 = np.rot90(w_ids[0], axes=(0,1))
    w_id_1 = np.rot90(w_ids[1], k=2, axes=(0,1))
    w_id_2 = np.rot90(w_ids[2], k=3, axes=(0,1))

    w_id = np.around((w_id_0 + w_id_1 + w_id_2) / 3.0).astype(np.uint8)

    i_w_id = w_id[:5, :50]
    h_w_id = w_id[5:, :50]
    o_w_id = w_id[5:, 50:53]

    return (
        val[i_w_id], 
        val[h_w_id], 
        val[o_w_id]
    )

def gif2arr(path, val):
    im = Image.open(path)
    return [png2arr(frame, val) for frame in iter_frames(im)]


class Game:
    def __init__(self):

        pygame.init()
        pygame.display.set_caption('*** Labirint 2 ***')
        self.sc = pygame.display.set_mode((WIDTH, HEIGHT))
        self.sc_map = pygame.Surface((WIDTH // MAP_SCALE, HEIGHT // MAP_SCALE))
        self.clock = pygame.time.Clock()
        self.drawing = Drawing(self.sc, self.sc_map)

        self.road_coords = set()
        self.finish_coords = set()
        self.wall_coord_list = list()
        self.world_map, self.collision_walls = get_map(TILE)
        for coord, signature in self.world_map.items():
            if signature == "1":
                self.wall_coord_list.append(coord)
            elif signature == "2":
                self.finish_coords.add(coord)
            elif signature == ".":
                self.road_coords.add(coord)

    def player_setup(self):

        weight_list = gif2arr(PDF_LOAD_PATH, VAL)

        self.player_list = []

        for weights in weight_list:
            player = Player(self.sc, self.collision_walls, self.finish_coords)
            color_r = np.random.randint(50, 220) # np.linspace(10, 200).astype(int)
            color_g = np.random.randint(50, 220) # np.linspace(10, 200).astype(int)
            color_b = np.random.randint(50, 220) # np.linspace(200, 10).astype(int)
            player.color = (color_r, color_g, color_b)
            player.init_angle = math.pi + (math.pi/2)
            player.rays = RayCast(self.world_map)
            player.brain = nn.Ganglion_numpy(weights)
            player.test_mode = True
            player.setup()
            self.player_list.append(player)

        print(f"Num of player: {len(self.player_list)}")

    def game_event(self):
        for player in self.player_list:
            player.movement()
            player.draw()
        
        for x, y in self.wall_coord_list:
            pygame.draw.rect(self.sc, WALL_COLOR_1, (x, y, TILE, TILE), 2)
        for x, y in self.finish_coords:
            pygame.draw.rect(self.sc, WALL_COLOR_2, (x, y, TILE, TILE), 2)

        # self.drawing.info(0, 0, 1, self.clock)

    def run(self):
        self.player_setup()
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    exit()
            self.sc.fill(BLACK)

            self.game_event()

            pygame.display.flip()
            self.clock.tick()

if __name__ == "__main__":
    game = Game()
    game.run()