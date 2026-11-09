"""Generate team kit textures for the left_team_kit / right_team_kit options.

    python -m grf_tools.make_kits

Kits follow the UV layout of GRF's built-in kits (1024 x 1024: shirt front
and back panels on top, the shorts below them, socks underneath). Writes
upstream/third_party/gfootball_engine/data/databases/default/images_teams/rl/.
Rebuild (./build.sh) afterwards so the engine data is reinstalled.
"""

from __future__ import annotations

import os

import cv2
import numpy as np

from grf_tools.make_ring_models import write_bmp_pixels

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "upstream", "third_party", "gfootball_engine", "data", "databases",
                    "default", "images_teams")
TEMPLATE = os.path.join(DATA, "primeradivision", "fcbarcelona_kit_02.bmp")
OUT = os.path.join(DATA, "rl")

# Bands of the texture (rows): shirt, shorts, sock tops, socks.
SHIRT, SHORTS, SOCK_TOP, SOCKS = (0, 420), (420, 582), (582, 644), (644, 770)

# name: shirt colours (one = plain, two = vertical stripes), shorts, sock top,
# socks. RGB.
KITS = {
    "barca_home": ([(0, 77, 152), (165, 0, 68)], (0, 35, 100), (165, 0, 68), (0, 35, 100)),
    "madrid_home": ([(245, 245, 245)], (245, 245, 245), (40, 40, 110), (245, 245, 245)),
    "madrid_away": ([(30, 30, 45)], (30, 30, 45), (210, 170, 60), (30, 30, 45)),
    "red": ([(200, 25, 30)], (245, 245, 245), (200, 25, 30), (200, 25, 30)),
    "sky": ([(110, 180, 235)], (245, 245, 245), (110, 180, 235), (245, 245, 245)),
}
STRIPE_PX = 48


def make_kit(shirt, shorts, sock_top, socks):
    template = cv2.imread(TEMPLATE, cv2.IMREAD_UNCHANGED)
    used = template[..., :3].sum(axis=2) > 0  # kit shape; black = unused
    image = np.zeros_like(template)
    image[..., 3] = 255
    cols = np.arange(image.shape[1])
    stripe = (cols % 512) // STRIPE_PX % len(shirt)  # per panel (front/back)
    for (top, bottom), colour in ((SHIRT, None), (SHORTS, shorts),
                                  (SOCK_TOP, sock_top), (SOCKS, socks)):
        if colour is None:
            band = np.array([shirt[i][::-1] for i in stripe], np.uint8)[None]
        else:
            band = np.array(colour[::-1], np.uint8)[None, None]
        rows = slice(top, bottom)
        image[rows, :, :3] = np.where(used[rows, :, None], band, 0)
    return image


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    for name, colours in KITS.items():
        image = make_kit(*colours)
        pixels = np.ascontiguousarray(image[::-1]).tobytes()  # BMP: bottom row first
        for number in (1, 2):  # the engine picks kit 01 or 02 per fixture
            write_bmp_pixels(os.path.join(OUT, "%s_kit_0%d.bmp" % (name, number)), pixels,
                             image.shape[1], image.shape[0])
        print("wrote", name)


if __name__ == "__main__":
    main()
