from typing import Optional

from .spaces.rgb import RGB


def print_fg_bg(bg: RGB, fg: RGB, text: str, error: Optional[str] = None, padding: int = 50):
    bg_val = bg.to_rgb255().values
    fg_val = fg.to_rgb255().values
    print(
        "\033[48;2;{};{};{}m".format(*bg_val)
        + "\033[38;2;{};{};{}m".format(*fg_val)
        + text.ljust(padding)
        + f"{bg_val}".rjust(13, " ")
        + "/"
        + f"{fg_val}".ljust(13, " ")
        + " | "
        + bg.to_hex().__str__()
        + "/"
        + fg.to_hex().__str__()
        + "\033[0m"
        + (f"\033[91m{error}\033[0m" if error is not None else "")
    )
