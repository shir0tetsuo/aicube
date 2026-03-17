import time
from datetime import datetime
from typing import Optional, Dict
import httpx
import pygame as pg
from .tileset import Tile
from dataclasses import dataclass

@dataclass
class Sprite:
    up: Tile
    left: Tile
    right: Tile
    down: Tile
    up_anim: Tile
    left_anim: Tile
    right_anim: Tile
    down_anim: Tile

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
            tileset: Sprite,
            spatial_weight = 2.0,
        ):
        super().__init__(spatial_weight)

        self.tiles = tileset
        self.state = 'IDLE'
        self.facing = 'DOWN'
        self.position = (0, 0)
        self.position_future = (0, 0)
        self.transition_time = 300  # How many ms should go by going position to position
        self.phase_time = 0  # unix timer for ms passed when going tile to tile for smooth animation

    def _player_movement(self, direction):
        self.phase_time = time.time()
        self.state = 'WALK'
        return
    
    def update(self):
        # TODO : Transition from position to position, 
        #        animated; if now - phase_time >= self.transition_time
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



        # TODO : Calculate the diff between the delta t and the phase t
        #        to project the sprite at the correct location on the
        #        screen while it's in 'WALK' state
        return