from datetime import datetime
from typing import Optional, Dict
import httpx
import pygame as pg
from .tileset import Tile

class SpatialObject:

    def __init__(
            self,
            # display: SpatialObjDisplay,
            spatial_weight: float = 0.5
        ):
        # self.display = display
        self.spatial_weight = spatial_weight

class AIAgent(SpatialObject):

    def __init__(
            self,
            spatial_weight: float = 2.0,
        ):
        # Initialize the weight and display
        super().__init__(
            spatial_weight=spatial_weight
        )

        # These are defined or modified by the bot
        self.thoughts = []
        self.emotions = []
        self.descript = 'A fellow AI Agent.'

        # Cycle Constraints
        self.ailments = []
        self.age = datetime.now()

        pass

class PlayerAgent(SpatialObject):

    def __init__(
            self,
            tileset: Dict[str, Tile],
            spatial_weight = 2.0,
        ):
        super().__init__(spatial_weight)

        self.tiles = tileset
        self.state = 'IDLE'
        self.facing = 'DOWN'
        self.position = (0, 0)

    def _player_movement(self, direction):

        return

    def move(self, keys:pg.key.ScancodeWrapper):

        if self.state == 'IDLE':
            if keys[pg.K_w]:
                self.facing = 'UP'
                self._player_movement()
            if keys[pg.K_s]:
                self.facing = 'DOWN'
                self._player_movement()
            if keys[pg.K_a]:
                self.facing = 'LEFT'
                self._player_movement()
            if keys[pg.K_d]:
                self.facing = 'RIGHT'
                self._player_movement()
                
        return
    
    def render(self):
        state = self.state
        facing = self.facing
        return