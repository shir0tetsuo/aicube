import re
from IPython.display import clear_output as cls
from typing import Callable
# cls()

def color_text(text: str, fg_hex: str = None, bg_hex: str = None, reset: bool = True) -> str:
    """
    Return text formatted with ANSI truecolor codes using hex colors.

    Parameters
    ----------
    text : str
        Text to print.
    fg_hex : str
        Foreground hex color (#RRGGBB).
    bg_hex : str
        Background hex color (#RRGGBB).
    reset : bool
        Reset terminal color after text.

    Returns
    -------
    str
    """

    def hex_to_rgb(h):
        h = h.lstrip('#')
        if not re.fullmatch(r"[0-9a-fA-F]{6}", h):
            raise ValueError(f"Invalid hex color: {h}")
        return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

    codes = []

    if fg_hex:
        r, g, b = hex_to_rgb(fg_hex)
        codes.append(f"\033[38;2;{r};{g};{b}m")

    if bg_hex:
        r, g, b = hex_to_rgb(bg_hex)
        codes.append(f"\033[48;2;{r};{g};{b}m")

    start = "".join(codes)
    end = "\033[0m" if reset else ""

    return f"{start}{text}{end}"

def cprint(text:str, fg=None, bg=None, end=None):
    print(color_text(text, fg, bg), end=end)

def Name(CLASS:Callable):
    return CLASS.__class__.__name__