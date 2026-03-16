from .objects import AIAgent, SpatialObject
from typing import List, Tuple, Dict, Union, Optional
from .screen import cls, cprint
import random

class Grid:

    def __init__(
        self,
        agents: Optional[List[AIAgent | SpatialObject]],
        grid_size: Tuple[int, int] = (9, 9),
        max_objects_in_space: int = 3
    ):

        self.width, self.height = grid_size

        # Maximum allotment of items per space
        self.mois = max_objects_in_space

        # Grid dictionary
        self.G: Dict[
            Tuple[int, int], 
                List[
                    AIAgent |
                    SpatialObject
                ]
            ] = {
            (x, y): []
            for x in range(self.width)
            for y in range(self.height)
        }

        # Add agents to the simulation
        if agents:
            for agent in agents:
                point = self.random_point(empty=True)
                self.G[point].append(agent)

        # Bounding Box
        self.bb = {
            'top': '┌' + ('─' * self.width) + '┐',
            'mid': '│',
            'btm': '└' + ('─' * self.width) + '┘'
        }

    # Check if there is available space by spatial weight
    def available_space(self, coords:Tuple[int, int]):
        mois = self.mois

        # Maximum of 9 objects at a point
        if len(self.G.get(coords, [])) >= 9:
            return (False, 0.0)

        for StackedObject in self.G.get(coords, []):
            if mois <= 0.0:
                break
            if hasattr(StackedObject, 'spatial_weight'):
                mois -= StackedObject.spatial_weight
        return ((True if (mois>=0.0) else False), mois)

    # Check if the coords are empty space
    def empty(self, coords:Tuple[int,int]):
        return not self.G.get(coords)

    # Obtain a random point from the grid
    def random_point(self, empty: bool = False) -> Tuple[int, int]:

        if not empty:
            return random.choice(list(self.G.keys()))

        empty_cells = [k for k, v in self.G.items() if not v]

        if not empty_cells:
            raise RuntimeError("No empty cells available")

        return random.choice(empty_cells)

    # Render Tick
    def update(self):
        cls(True)  # clear terminal

        # print top border
        print(self.bb['top'])

        for y in range(self.height):

            print(self.bb['mid'], end='')

            for x in range(self.width):

                cell = self.G[(x, y)]

                if not cell:
                    # empty space
                    print(' ', end='')
                    continue

                # number of objects in this cell
                count = len(cell)

                # find the heaviest object (highest spatial_weight)
                heaviest = max(cell, key=lambda o: o.spatial_weight)

                # get the foreground color from display
                fg = getattr(heaviest.display, 'fg', '#000000')

                # print count in color
                cprint(str(count), color=fg, end='')

            print(self.bb['mid'])

        # print bottom border
        print(self.bb['btm'])