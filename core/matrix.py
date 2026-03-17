import pygame
import sys
import math
from .grid import tileset, Grid
from .tileset import Tile
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
                up_anim=Tile(tileset, pointer=[('AF', 100), ('C7', 100)]),
                left_anim=Tile(tileset, pointer=[('B7', 100), ('CF', 100)]),
                right_anim=Tile(tileset, pointer=[('B7', 100), ('CF', 100)], flip_horizontal=True),
                down_anim=Tile(tileset, pointer=[('A7', 100), ('BF', 100)])
            ),
            spatial_weight=2.0
        )

        self.grids = [
            Grid(agents=[self.Player])
        ]

        pass

    def handleCoreInterrupts(self, keys:pygame.key.ScancodeWrapper):
        for event in pygame.event.get():
            if (event.type == pygame.QUIT) or keys[pygame.K_ESCAPE]:
                self.running = False
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                print(event)

            if (event.type == pygame.VIDEORESIZE):
                self.WH = event.size

    def Tick(self, t:int=20):
        self.clock.tick(24)
        dt = self.clock.get_time() #/ 1000.0
        keys = pygame.key.get_pressed()
        events = self.handleCoreInterrupts(keys)
        return dt, keys, events
    
    def LOOP(self):
        while self.running:
            self.screen.fill((10, 10, 20))
            dt, keys, events = self.Tick(24)

            # Update all grids
            for grid in self.grids:
                grid.update(keys, dt)

                player_coords = grid.find_player()

                if player_coords:
                    player = next(
                        i for i in grid.G[player_coords]
                        if isinstance(i, PlayerAgent)
                    )
                    render_x, render_y = player.render_position
                    sprite = player.render_sprite

                    # Rendering
                    grid.camera_projections(
                        (render_x, render_y),
                        sprite,
                        self.screen
                    )

            pygame.display.flip()