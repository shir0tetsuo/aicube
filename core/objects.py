from datetime import datetime
from typing import Optional
import httpx

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

    def __init__(self, spatial_weight = 0.5):
        super().__init__(spatial_weight)

    