import time
from datetime import datetime
from typing import Optional, Dict, List
import httpx
import pygame as pg
from .tileset import Tile
from dataclasses import dataclass
from .terminal import cprint, Name

# TODO | NOTE : If we aren't done the walk cycle, continue
#               with the walk animation, no jerking/pause.
# TODO : Face a direction without doing a movement, maybe using dt
# TODO : Facing action (E?) and UI

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

    def __repr__(self):
        objId = hex(id(self))
        short = '0x..'+objId[-4:]
        return f'<{short}:{Name(self)}>'

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
            sprite: Sprite,
            spatial_weight = 2.0,
        ):
        super().__init__(spatial_weight)

        self.sprite = sprite
        self.state = 'IDLE'
        self.facing = 'DOWN'
        self.position = (0, 0)
        self.position_start = (0, 0)
        self.position_future = (0, 0)
        self.transition_time = 300  # How many ms should go by going position to position
        self.phase_time = 0         # unix timer for ms passed when going tile to tile for smooth animation
        self.phase_elapsed = 0.0    # unix timer +=dt float
        self.input_threshold = 50
        # self.next_direction = None
        # self.passables = { d: False for d in ['UP','DOWN','LEFT','RIGHT'] }
        
        # Rendering
        self.render_position = (0, 0)
        self.render_sprite = self.sprite.down

        # The graphic is passed for rendering
        self.render_state = {
            ('IDLE', 'UP'): self.sprite.up,
            ('IDLE', 'DOWN'): self.sprite.down,
            ('IDLE', 'LEFT'): self.sprite.left,
            ('IDLE', 'RIGHT'): self.sprite.right,
            ('WALK', 'UP'): self.sprite.up_anim,
            ('WALK', 'DOWN'): self.sprite.down_anim,
            ('WALK', 'LEFT'): self.sprite.left_anim,
            ('WALK', 'RIGHT'): self.sprite.right_anim,
        }

    def _player_movement(
            self, 
            # direction:str = None
        ):
        '''
        Set player movement to `'WALK'` and proceed
        based on relative position.
        '''
        
        self.phase_elapsed = 0.0
        self.state = 'WALK'
        self.position_start = self.position
        
        # if direction:
        #     self.facing = direction

        facing = self.facing
        x,y = self.position

        self.position_future = {
            'LEFT': (x-1, y),
            'RIGHT': (x+1, y),
            'UP': (x, y-1),
            'DOWN': (x, y+1)
        }.get(facing, (x, y))

        return
    
    # The position is passed for rendering
    def update(
            self, 
            dt:float, 
            # keys:pg.key.ScancodeWrapper
        ):
        # TODO : Ledge Detection


        # LERP Position Calculation

        if self.state != 'WALK':
            return self.position

        # accumulate ms
        self.phase_elapsed += dt

        # progress 0 → 1
        t = min(self.phase_elapsed / self.transition_time, 1.0)

        # Smoothstep easing
        t = t * t * (3 - 2 * t)

        x0, y0 = self.position_start
        x1, y1 = self.position_future

        # Linear interpolation
        xf = x0 + (x1 - x0) * t
        yf = y0 + (y1 - y0) * t

        # Quantize to nearest 1/8
        def quantize(v):
            return round(v * 8) / 8

        xf = quantize(xf)
        yf = quantize(yf)

        # If movement finished
        if t >= 1.0:
            self.position = self.position_future
            self.state = 'IDLE'
            return self.position

        return (xf, yf)

    def move(
            self, 
            dt: float,
            keys: pg.key.ScancodeWrapper, 
            passables: Dict[str, bool] = { d: False for d in ['UP','DOWN','LEFT','RIGHT'] }
        ):

        # self.passables = passables

        direction = None

        if keys[pg.K_w]:
            direction = 'UP'
        elif keys[pg.K_s]:
            direction = 'DOWN'
        elif keys[pg.K_a]:
            direction = 'LEFT'
        elif keys[pg.K_d]:
            direction = 'RIGHT'

        # No key pressed → reset buffer
        if direction is None:
            self.input_hold_time = 0.0
            self.input_direction = None
            return

        # New key press → reset timer
        if self.input_direction != direction:
            self.input_direction = direction
            self.input_hold_time = 0.0

            # Always update facing immediately
            self.facing = direction
            return

        # Accumulate hold time
        self.input_hold_time += dt

        # If still idle, decide what to do
        if self.state == 'IDLE':
            # Face instantly (already done above, but safe)
            self.facing = direction

            # Only move after threshold
            if self.input_hold_time >= self.input_threshold:
                if passables.get(direction, False):
                    self._player_movement()

        if keys[pg.K_SPACE]:
            cprint(f'{repr(self)}: {self.position}', fg="#0DE4D9", bg="#202020")

        self.next_direction = direction