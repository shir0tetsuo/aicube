from PIL import Image
import time
from typing import Optional, Literal


# Global tile cache
_TILE_CACHE: dict[tuple[int, str], Image.Image] = {}


class Tile:

    def __init__(
        self,
        tileset: Image.Image,
        pointer: list[tuple[str, int]],  # hex_id, ms
        tile_size: int = 32,
        blank_quads: list[bool] | None = None,
        water_quads: bool = False,
        water_pointer: list[tuple[str, int]] = [
            ("5C", 200),
            ("5D", 200),
            ("5E", 200),
            ("5F", 200)
        ],
        name: Optional[str] = None,
        collision: Literal['passable', 'impassable', 'liquid', 'ledge'] = 'passable'
    ):
        """
        Tile supporting static or animated frames and optional water replacement.

        pointer format:
            [("1F", 200), ("20", 200)]

        blank_quad order:
            `[Q1, Q2, Q3, Q4]`

        ```
        Q1 | Q2
        -------
        Q3 | Q4
        ```
        """

        self.name = name
        self.collision = collision
        self.tileset = tileset
        self.tile_size = tile_size
        self.blank_quads = blank_quads or [False, False, False, False]

        self.render_water = water_quads
        self.water_pointer = water_pointer or []

        self.frames = []
        self.durations = []

        for hex_id, duration in pointer:
            tile = self._get_tile_by_hex(hex_id)
            self.frames.append(tile)
            self.durations.append(duration)

        self.frame_count = len(self.frames)

        self.start_time = time.time()

        # Prepare water frames if enabled
        self.water_frames = []
        self.water_durations = []

        if self.render_water and self.water_pointer:
            for hex_id, duration in self.water_pointer:
                tile = self._get_tile_by_hex(hex_id)
                self.water_frames.append(tile)
                self.water_durations.append(duration)

    def _get_cached_base_tile(self, hex_id: str) -> Image.Image:
        """
        Return cached tile from tileset without quad masking.
        """

        key = (id(self.tileset), hex_id)

        if key in _TILE_CACHE:
            return _TILE_CACHE[key]

        tile_number = int(hex_id, 16)

        tiles_per_row = self.tileset.width // self.tile_size

        y = tile_number // tiles_per_row
        x = tile_number % tiles_per_row

        left = x * self.tile_size
        upper = y * self.tile_size
        right = left + self.tile_size
        lower = upper + self.tile_size

        tile = self.tileset.crop((left, upper, right, lower)).convert("RGBA")

        _TILE_CACHE[key] = tile

        return tile

    def _get_tile_by_hex(self, hex_id: str) -> Image.Image:
        """
        Get tile and apply quad masking if required.
        """

        base = self._get_cached_base_tile(hex_id)

        if not any(self.blank_quads):
            return base

        tile = base.copy()

        quad_w = self.tile_size // 2
        quad_h = self.tile_size // 2

        pixels = tile.load()

        quads = [
            (0, 0, quad_w, quad_h),
            (quad_w, 0, self.tile_size, quad_h),
            (0, quad_h, quad_w, self.tile_size),
            (quad_w, quad_h, self.tile_size, self.tile_size),
        ]

        for i, blank in enumerate(self.blank_quads):
            if blank:
                x1, y1, x2, y2 = quads[i]
                for py in range(y1, y2):
                    for px in range(x1, x2):
                        r, g, b, a = pixels[px, py]
                        pixels[px, py] = (r, g, b, 0)

        return tile

    def _get_animated_frame(self, frames, durations):

        if len(frames) == 1:
            return frames[0]

        elapsed_ms = (time.time() - self.start_time) * 1000
        total_duration = sum(durations)

        elapsed_ms %= total_duration

        total = 0
        for frame, duration in zip(frames, durations):
            total += duration
            if elapsed_ms <= total:
                return frame

        return frames[-1]

    def get_frame(self) -> Image.Image:

        base = self._get_animated_frame(self.frames, self.durations)

        if not self.render_water or not any(self.blank_quads):
            return base

        water = self._get_animated_frame(self.water_frames, self.water_durations)

        result = water.copy()

        quad_w = self.tile_size // 2
        quad_h = self.tile_size // 2

        quads = [
            (0, 0, quad_w, quad_h),
            (quad_w, 0, self.tile_size, quad_h),
            (0, quad_h, quad_w, self.tile_size),
            (quad_w, quad_h, self.tile_size, self.tile_size),
        ]

        for i, blank in enumerate(self.blank_quads):

            if blank:
                continue

            x1, y1, x2, y2 = quads[i]

            region = base.crop((x1, y1, x2, y2))
            result.paste(region, (x1, y1), region)

        return result