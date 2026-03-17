from .objects import AIAgent, SpatialObject, PlayerAgent
from typing import List, Tuple, Dict, Union, Optional
from .terminal import cls, cprint
from .tileset import Tile
import random
from pathlib import Path
import os
import json
from PIL import Image
import pygame as pg
import threading

images_root = Path(os.path.join(Path(__file__).parent.parent.resolve(), 'images'))
maps_root = Path(os.path.join(Path(__file__).parent.resolve(), 'maps'))
maps = {
    int(str(m.stem).strip('map')) :
    os.path.join(str(maps_root), m)
    for m in maps_root.iterdir()
}

# Load the tileset
tileset = Image.open(os.path.join(images_root, 'PkCrystalTiles.png'))

class Grid:

    tiles = {

        # (SCREEN FILL)
        -1: Tile(tileset, pointer=[('5C', 0)], blank_quads=[True, True, True, True], collision='impassable'),
        # WATER
        # 0: Tile(tileset, pointer=[('5C', 0)], blank_quads=[False, True, True, False], water_quads=True, collision='liquid'),
        0: Tile(tileset, pointer=[
                ("5C", 1000),
                ("5D", 1000),
                ("5E", 1000),
                ("5F", 1000)
            ], collision='liquid'),
        
        # OPEN SPACE
        1: Tile(tileset, pointer=[('28', 0)]),
        2: Tile(tileset, pointer=[('29', 0)]),
        3: Tile(tileset, pointer=[('35', 0)]),
        4: Tile(tileset, pointer=[('1', 0)]),

        # GRASS
        5: Tile(tileset, pointer=[('0', 0)]),
        6: Tile(tileset, pointer=[('30', 0)])

    }

    map_lock = threading.RLock()

    def __init__(
        self,
        agents: Optional[List[AIAgent | SpatialObject | PlayerAgent]],
        map_number = 0,
        max_objects_in_space: float = 3.0,
        tile_size = 32  # pixels as width and height per tile
    ):
        global images_root, maps

        # Load the map and sprite data
        with self.map_lock:
            with open(maps[map_number], 'r', encoding='utf-8') as m:
                self.map_data = json.load(m)
                
                points = [tuple(cell["point"]) for cell in self.map_data["map"]]
                xs, ys = zip(*points)
                # Infer width and height
                self.width = max(xs) + 1
                self.height = max(ys) + 1

        # Maximum allotment of items per space
        self.mois = max_objects_in_space

        # Grid dictionary of objects, entities
        self.G: Dict[
            Tuple[int, int], 
                List[
                    AIAgent |
                    SpatialObject |
                    PlayerAgent
                ]
            ] = {
            (x, y): []
            for x in range(self.width)
            for y in range(self.height)
        }

        self.tile_size = tile_size
        # Tile grid
        self.GTiles: Dict[Tuple[int, int], Tile] = {
            (x, y): self.tiles[-1]
            for x in range(self.width)
            for y in range(self.height)
        }

        # Decode tile data from map
        for cell in self.map_data.get("map", []):

            point = tuple(cell.get("point", (0, 0)))

            tile_id = cell.get("tile", -1)

            if tile_id not in self.tiles:
                tile_id = -1

            if point in self.GTiles:
                self.GTiles[point] = self.tiles[tile_id]

        # Add agents to the simulation
        # TODO : add entities (objects) from map data
        if agents:
            for agent in agents:
                point = self.random_point(empty=True)
                self.G[point].append(agent)

    # Check what type of collision the coords tile has
    def collision_space(self, coords:Tuple[int, int]):
        GTile = self.GTiles.get(coords)
        if GTile:
            return GTile.collision
        else:
            return 'impassable'

    # Check if there is available space by spatial weight
    def available_space(self, coords:Tuple[int, int]):
        mois = self.mois

        # Point tile must be passable
        if self.collision_space(coords) != 'passable':
            return (False, 0.0)

        # Maximum of 9 objects at a point
        if len(self.G.get(coords, [])) >= 9:
            return (False, 0.0)

        for StackedObject in self.G.get(coords, []):
            if mois <= 0.0:
                break
            if hasattr(StackedObject, 'spatial_weight'):
                mois -= StackedObject.spatial_weight
        return ((True if (mois>=0.0) else False), mois)
    
    # Has Space (for more objects on stack, like item objects)
    def has_space(self, coords:Tuple[int, int], weight:float):
        available, mois = self.available_space(coords=coords)
        if not available:
            return False
        if (mois - weight) >= 0.0:
            return True
        return False

    # Check if the coords are empty space
    def empty_space(self, coords:Tuple[int,int]):
        return not self.G.get(coords)

    # Obtain a random point from the grid
    def random_point(self, empty: bool = False, need_passable: bool = True) -> Tuple[int, int]:
        """
        Returns a random point in the grid.
        If `empty=True`, only considers empty cells.
        If `need_passable=True`, only considers tiles with `collision='passable'`.
        """

        candidates = list(self.G.keys())

        if empty:
            candidates = [k for k, v in self.G.items() if not v]

        if need_passable:
            candidates = [k for k in candidates if self.collision_space(k) == 'passable']

        if not candidates:
            raise RuntimeError("No suitable cells available (empty/passable)")

        return random.choice(candidates)
    
    def find_player(self) -> Optional[Tuple[int, int]]:
        """
        Returns the coordinates of the first PlayerAgent in the grid,
        or None if no PlayerAgent is present.
        """
        for coords, cell in self.G.items():
            for obj in cell:
                if isinstance(obj, PlayerAgent):
                    return coords
        return None

    # Might be useful later ...
    def find_all_players(self) -> List[Tuple[int, int]]:
        """
        Returns a list of coordinates of all PlayerAgents in the grid.
        """
        players = []
        for coords, cell in self.G.items():
            if any(isinstance(obj, PlayerAgent) for obj in cell):
                players.append(coords)
        return players
    
    def camera_projections(
        self,
        center: Tuple[float, float],
        sprite: Tile,
        screen: pg.Surface,
        scale: int = 2,
        min_tiles: int = 10
    ):
        """
        Render a camera-centered view onto a pygame screen.
        """

        # TODO : Fill empty space with an impassable tile

        screen_w, screen_h = screen.get_size()
        tile_size = self.tile_size * scale

        # --- Determine how many tiles fit on screen ---
        tiles_x = max(min_tiles, screen_w // tile_size + 2)
        tiles_y = max(min_tiles, screen_h // tile_size + 2)

        cx, cy = center

        # --- Determine visible region (float-safe) ---
        x1 = int(cx - tiles_x // 2)
        y1 = int(cy - tiles_y // 2)
        x2 = x1 + tiles_x
        y2 = y1 + tiles_y

        # --- Clamp to grid bounds ---
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(self.width, x2)
        y2 = min(self.height, y2)

        # --- Render tiles ---
        for y in range(y1, y2):
            for x in range(x1, x2):

                tile = self.GTiles.get((x, y), self.tiles[-1])
                frame = tile.get_frame()

                # Convert to pygame surface
                mode = frame.mode
                size = frame.size
                data = frame.tobytes()

                surf = pg.image.fromstring(data, size, mode)

                if scale != 1:
                    surf = pg.transform.scale(surf, (tile_size, tile_size))

                # Screen position
                sx = int((x - cx) * tile_size + screen_w // 2)
                sy = int((y - cy) * tile_size + screen_h // 2)

                screen.blit(surf, (sx, sy))

        # --- Render entities (player for now) ---
        px, py = center

        if sprite:
            frame = sprite.get_frame()

            mode = frame.mode
            size = frame.size
            data = frame.tobytes()

            surf = pg.image.fromstring(data, size, mode)

            if scale != 1:
                surf = pg.transform.scale(surf, (tile_size, tile_size))

            sx = int((px - cx) * tile_size + screen_w // 2)
            sy = int((py - cy) * tile_size + screen_h // 2)

            screen.blit(surf, (sx, sy))
    
    def update(self, keys:pg.key.ScancodeWrapper, dt:float):
        # Obtain the player from the grid.
        player_coords = self.find_player()

        if player_coords:
            player: PlayerAgent = next(
                (i for i in self.G[player_coords] if isinstance(i, PlayerAgent)),
                None  # optional default if not found
            )
            if player is None:
                return # no player found, currently fine if this is the only update
            
            x,y=player_coords
            w = player.spatial_weight
        
            # Perform player movement based on spatial weights
            passable = {
                'LEFT':  self.has_space((x-1, y), w), # left
                'RIGHT': self.has_space((x+1, y), w), # right
                'UP':    self.has_space((x, y-1), w), # up
                'DOWN':  self.has_space((x, y+1), w)  # down
            }
            player.move(dt, keys, passable)

            # NOTE : This data can be used to render the sprite.,
            #        and the "camera".
            render_x, render_y = player.update(dt)

            sprite = player.render_state.get((player.state, player.facing))
            # If animated tile supports frames:
            # if player.state == 'WALK':
            #     sprite_frame = sprite.get_frame(player.phase_elapsed)
            # else:
            #     sprite_frame = sprite.get_frame()

            player.render_position = (render_x, render_y)
            player.render_sprite = sprite

            # self.camera_projections(screen) ...

            # Move the player on the grid if position is
            # not equal to the player's position
            if isinstance(render_x, int) and isinstance(render_y, int):
                if (render_x != x) or (render_y != y):
                    if isinstance(self.G.get((render_x, render_y), None), list):
                        self.G[player_coords].remove(player)
                        self.G[(render_x, render_y)].append(player)
                    else:
                        # Safety
                        player.position = player_coords
                        player.position_start = player_coords
                        player.position_future = player_coords
                        player.render_position = player_coords
                        cprint(f'ERROR: Grid does not contain position {render_x}, {render_y}.', fg='#FFFFFF', bg="#B60000")
        return

    def render_map(
        self,
        scale: int = 1,
        region: Optional[Tuple[int, int, int, int]] = None
    ) -> Image.Image:
        """
        Render the map to a PIL image.

        scale  : integer scaling factor
        region : optional (x1,y1,x2,y2) subset of grid
        """

        # Determine tile size dynamically
        sample_tile = next(iter(self.GTiles.values()))
        tile_size = sample_tile.tile_size

        # Region selection (for cameras later)
        if region:
            x1, y1, x2, y2 = region
        else:
            x1, y1 = 0, 0
            x2, y2 = self.width, self.height

        width = x2 - x1
        height = y2 - y1

        img_w = width * tile_size
        img_h = height * tile_size

        canvas = Image.new("RGBA", (img_w, img_h))

        for y in range(y1, y2):
            for x in range(x1, x2):

                tile = self.GTiles.get((x, y), self.tiles[-1])

                frame = tile.get_frame()

                px = (x - x1) * tile_size
                py = (y - y1) * tile_size

                canvas.paste(frame, (px, py), frame)

        if scale != 1:
            canvas = canvas.resize(
                (canvas.width * scale, canvas.height * scale),
                Image.NEAREST
            )

        return canvas        

    # Render Tick
    # def render(self):
        
        # cls(True)  # clear terminal

        # # print top border
        # print(self.bb['top'])

        # for y in range(self.height):

        #     print(self.bb['mid'], end='')

        #     for x in range(self.width):

        #         cell = self.G[(x, y)]

        #         if not cell:
        #             # empty space
        #             print(' ', end='')
        #             continue

        #         # number of objects in this cell
        #         count = len(cell)

        #         # find the heaviest object (highest spatial_weight)
        #         heaviest = max(cell, key=lambda o: o.spatial_weight)

        #         # get the foreground color from display
        #         fg = getattr(heaviest.display, 'fg', '#000000')

        #         # print count in color
        #         cprint(str(count), color=fg, end='')

        #     print(self.bb['mid'])

        # # print bottom border
        # print(self.bb['btm'])

