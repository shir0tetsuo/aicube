import pygame
import sys
import math
from .grid import tileset, Grid
from .tileset import Tile
from .terminal import cprint
from .objects import PlayerAgent, Sprite

class Matrix:
    def __init__(self):
        pygame.init()
        self.WH = (800,600)
        self.screen = pygame.display.set_mode(self.WH, pygame.RESIZABLE)
        self.font = pygame.font.Font(None, 20)
        self.clock = pygame.time.Clock()

        self.running = True

        self.Player = PlayerAgent(
            Sprite(
                up=Tile(tileset, pointer=[('AF', 0)]),
                left=Tile(tileset, pointer=[('B7', 0)]),
                right=Tile(tileset, pointer=[('B7', 0)], flip_horizontal=True),
                down=Tile(tileset, pointer=[('A7', 0)]),
                up_anim=Tile(tileset, pointer=[('AF', 200), ('C7', 200)]),
                left_anim=Tile(tileset, pointer=[('B7', 200), ('CF', 200)]),
                right_anim=Tile(tileset, pointer=[('B7', 200), ('CF', 200)], flip_horizontal=True),
                down_anim=Tile(tileset, pointer=[('A7', 200), ('BF', 200)])
            ),
            spatial_weight=2.0
        )

        self.grids = [
            Grid(agents=[self.Player])
        ]

        pass

    def handleCoreInterrupts(self, keys:pygame.key.ScancodeWrapper):
        events = pygame.event.get()
        for event in events:
            if (event.type == pygame.QUIT) or keys[pygame.K_ESCAPE]:
                self.running = False
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                cprint(repr(event), "#3A3A3A", "#757575")

            if (event.type == pygame.VIDEORESIZE):
                self.WH = event.size

        return events

    def Tick(self, t:int=20):
        self.clock.tick(24)
        dt = self.clock.get_time() #/ 1000.0
        keys = pygame.key.get_pressed()
        events = self.handleCoreInterrupts(keys)
        return dt, keys, events
    
    def LOOP(self):
        while self.running:
            self.screen.fill((38, 38, 38))
            dt, keys, events = self.Tick(24)

            # Update all grids
            for grid in self.grids:
                grid.update_player(keys, dt, self.screen)

            pygame.display.flip()