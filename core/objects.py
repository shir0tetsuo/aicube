from dataclasses import dataclass
from datetime import datetime
from typing import Optional
import httpx

@dataclass
class SpatialObjDisplay:
    char: Optional[str] = ' '
    fg: Optional[str] = '#000000'
    bg: Optional[str] = '#FFFFFF'

class SpatialObject:

    def __init__(
            self,
            display: SpatialObjDisplay,
            spatial_weight: float = 0.5
        ):
        self.display = display
        self.spatial_weight = spatial_weight

class AIAgent(SpatialObject):

    def __init__(
            self,
            spatial_weight: float = 1.5,
            char = 'A', fg = "#A6FF00", bg = '#FFFFFF'
        ):
        # Initialize the weight and display
        super().__init__(
            display=SpatialObjDisplay(
                char=char, fg=fg, bg=bg
            ),
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