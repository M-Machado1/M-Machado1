"""
Subtle 2D 'living illustration' animation for a flat banner image.

The banner is a single flattened bitmap (no layers), so the character is animated
with smooth displacement-field warps that are *masked* to the character only
(hair, head, cables, ribbons). Everything else (text, background, halftone,
shapes) is copied from the original image and stays pixel-identical.

Loop: every motion term is a periodic function f(t) and the warp uses
f(t) - f(0), so frame 0 is exactly the original and frame N == frame 0.

Usage:
    python scripts/animate_banner.py [source_image] [output_gif]

Defaults:
    source  -> assets/murilo-banner-animated-1024.gif
    output  -> assets/banner.gif
"""
import sys
import os
import numpy as np
import cv2
from PIL import Image

# ---------------------------------------------------------------- paths
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "assets", "murilo-banner-animated-1024.gif")
DST = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "assets", "banner.gif")

# ---------------------------------------------------------------- tunables
HAIR_AMP = 2.5       # px at the hair tips (scaled 0.5x from 2048x768 source)
HEAD_FLOAT = 0.7     # px vertical bob (scaled 0.5x)
HEAD_DRIFT = 0.35    # px horizontal drift (scaled 0.5x)
HEAD_TILT_DEG = 0.22 # degrees (angle - unchanged)
RIBBON_AMP = 1.75    # px at the ribbon tips (scaled 0.5x)
CABLE_AMP = 0.8      # px (scaled 0.5x)
# ------------------------------------------------------------------------

img = cv2.imread(SRC, cv2.IMREAD_COLOR)            # BGR
if img is None:
    raise FileNotFoundError(f"Cannot read source image: {SRC}")
H, W = img.shape[:2]

# Coordinate system scaled from the original 2048x768 to the actual 1024x384 banner.
X0, Y0 = 400, 30                                   # ROI that contains the character + accessories
roi = img[Y0:H, X0:W].copy()
rh, rw = roi.shape[:2]
gx, gy = np.meshgrid(np.arange(X0, X0 + rw, dtype=np.float32),
                     np.arange(Y0, Y0 + rh, dtype=np.float32))   # full-image coords
Rc, Gc, Bc = roi[..., 2].astype(np.int32), roi[..., 1].astype(np.int32), roi[..., 0].astype(np.int32)
mx = np.maximum(np.maximum(Rc, Gc), Bc)
mn = np.minimum(np.minimum(Rc, Gc), Bc)


def poly_mask(pts):
    m = np.zeros((rh, rw), np.uint8)
    p = (np.array(pts, np.float32) - [X0, Y0]).astype(np.int32)
    cv2.fillPoly(m, [p], 255)
    return m.astype(np.float32) / 255.0


def blur(m, s):
    return cv2.GaussianBlur(m, (0, 0), s)


def ellipse_mask(cx, cy, ax, ay, feather):
    m = np.zeros((rh, rw), np.uint8)
    cv2.ellipse(m, (int(cx - X0), int(cy - Y0)), (int(ax), int(ay)), 0, 0, 360, 255, -1)
    m = blur(m.astype(np.float32) / 255.0, feather)
    return np.clip(m * 1.6, 0, 1)


def smooth_noise(seed, sigma):
    rng = np.random.RandomState(seed)
    n = rng.randn(rh, rw).astype(np.float32)
    n = cv2.GaussianBlur(n, (0, 0), sigma)
    n /= (np.abs(n).max() + 1e-6)
    return n  # roughly [-1, 1]


# ------------------------------------------------------------------ masks
# --- hair: generous polygon x "is hair colour" (excludes white skin and the dark-navy background)
hair_poly = poly_mask([(790, 184), (797, 180), (825, 166), (855, 153), (890, 140), (925, 136),
                       (975, 133), (1024, 129), (1024, 300), (1015, 312), (995, 334), (967, 353), (935, 351),
                       (900, 343), (850, 335), (826, 338), (783, 333), (750, 323), (720, 304),
                       (709, 298), (725, 291), (760, 287), (780, 270), (800, 250), (820, 230),
                       (850, 220), (860, 205), (850, 190), (820, 186)])
not_skin = (mn < 150).astype(np.float32)
not_navy = (Bc >= 72).astype(np.float32)
hair_col = blur(not_skin * not_navy, 1.25)
hair_w = np.clip(blur(hair_poly, 3) * hair_col, 0, 1)
hair_w = blur(hair_w, 1.5)

# protect the eyes (and their lash strokes) so the gaze / expression is untouched
eye_protect = np.clip(1.0 - np.maximum(ellipse_mask(906, 196, 49, 41, 11),
                                       ellipse_mask(851, 268, 31, 19, 9)), 0, 1)
hair_w *= eye_protect

# keep the picture border and the navy backdrop at the top completely still
edge = np.clip((W - 8 - gx) / 18.0, 0, 1)
hair_w *= edge

# --- head envelope (very small rigid motion with a wide, smooth falloff)
head_env = np.exp(-(((gx - 825) / 145.0) ** 2 + ((gy - 190) / 150.0) ** 2))
head_env *= edge
head_env *= np.clip((332 - gy) / 35.0, 0, 1)            # nothing moves below the chin / lower hair
head_env *= np.clip((gx - 650) / 70.0, 0, 1)          # keep the left halftone band still
# static backdrop (dark navy + halftone + cyan stripes above the head) never moves
static_bg = np.maximum(
    poly_mask([(592, 30), (830, 30), (830, 56), (800, 56), (780, 64), (750, 70), (720, 68), (650, 56)]),
    poly_mask([(850, 30), (1024, 30), (1024, 129), (975, 133), (925, 136), (890, 140), (880, 127), (872, 60)]))
head_env *= 1.0 - np.clip(blur(static_bg, 2.5) * 1.3, 0, 1)

# --- black ribbons (loose accessories)
rib_zone = poly_mask([(413, 50), (672, 50), (672, 215), (492, 215), (492, 107), (413, 107)])
rib = ((mx < 75).astype(np.float32)) * rib_zone
rib = cv2.dilate(rib, np.ones((5, 5), np.uint8))
rib_w = np.clip(blur(rib, 2.5) * 1.5, 0, 1)
rib_w *= np.clip((gx - 413) / 3.0, 0, 1)

# --- blue headphone cables: line-like dark-blue strokes on light surfaces
cab_zone = blur(poly_mask([(622, 59), (806, 59), (806, 125), (780, 150), (735, 235), (625, 235)]), 2)
cab_col = ((Bc > 110) & (Rc < 125) & (Gc > 38) & ((Bc - Rc) > 60)).astype(np.float32)
gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).astype(np.float32)
local = cv2.GaussianBlur(gray, (0, 0), 7)
cab_dark = ((local - gray) > 35).astype(np.float32)
cab = cab_col * cab_dark * cab_zone
cab = cv2.dilate(cab, np.ones((4, 4), np.uint8))
cab_w = np.clip(blur(cab, 2.5) * 1.4, 0, 1)
cab_w *= eye_protect

# ------------------------------------------------------- spatial phase maps
n1 = smooth_noise(3, 13)       # per-lock phase offsets (independent locks)
n2 = smooth_noise(11, 7)      # finer flutter
n3 = smooth_noise(29, 20)
HR = np.array([860.0, 125.0], np.float32)      # hair "root" (top of the head)
rx, ry = gx - HR[0], gy - HR[1]
rdist = np.sqrt(rx ** 2 + ry ** 2) + 1e-3
tip = np.clip(rdist / 260.0, 0, 1) ** 1.4        # tips move more than roots
# tangential (sway) + a little radial direction
tx, ty = -ry / rdist, rx / rdist
rdx, rdy = rx / rdist, ry / rdist
hair_dirx = tx + 0.30 * rdx
hair_diry = ty + 0.30 * rdy

KN = np.array([631.0, 124.0], np.float32)     # ribbon knot
kx, ky = gx - KN[0], gy - KN[1]
kd = np.sqrt(kx ** 2 + ky ** 2)
rib_tip = np.clip(kd / 215.0, 0, 1) ** 1.2

cx, cy = 850.0, 175.0                           # head rotation centre
TWO_PI = 2 * np.pi


def fields(t):
    """Raw (un-zeroed) displacement fields for normalised time t in [0,1)."""
    a = TWO_PI * t
    # --- hair: primary sway (k=1, delayed with distance + per-lock offsets) and flutter (k=2)
    s1 = np.sin(a - rdist / 75.0 - 1.7 * n1 - 0.9 * n3)
    s2 = np.sin(2 * a - rdist / 45.0 - 2.4 * n2 + 1.3)
    hair_mag = HAIR_AMP * tip * (0.80 * s1 + 0.28 * s2) * hair_w
    # extra secondary motion that lags the head bob
    lag = np.sin(a - 0.9) * 1.1 * hair_w
    dx = hair_mag * hair_dirx
    dy = hair_mag * hair_diry + lag

    # --- head: float + tilt (tiny)
    bob = np.sin(a + 0.4)
    drift = np.sin(a + 1.2)
    th = np.deg2rad(HEAD_TILT_DEG) * np.sin(a + 0.7)
    dx += head_env * (HEAD_DRIFT * drift - th * (gy - cy))
    dy += head_env * (HEAD_FLOAT * bob + th * (gx - cx))

    # --- ribbons (delayed, whip-like)
    rs = np.sin(a - kd / 60.0 - 0.6) + 0.30 * np.sin(2 * a - kd / 35.0)
    rm = RIBBON_AMP * rib_tip * rs * rib_w
    dx += 0.30 * rm
    dy += rm

    # --- cables: slight dangle
    cs = np.sin(a - 0.8 - (gx - 625) / 70.0) + 0.25 * np.sin(2 * a - (gx - 625) / 45.0)
    dy += CABLE_AMP * cs * cab_w
    dx += 0.25 * CABLE_AMP * np.sin(a - 1.4) * cab_w
    return dx, dy


F0 = fields(0.0)


def render(t):
    dx, dy = fields(t)
    dx = dx - F0[0]
    dy = dy - F0[1]
    mapx = (gx - X0 + dx).astype(np.float32)
    mapy = (gy - Y0 + dy).astype(np.float32)
    warped = cv2.remap(roi, mapx, mapy, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    moved = (np.abs(dx) + np.abs(dy)) > 0.06
    out = img.copy()
    sub = out[Y0:H, X0:W]
    sub[moved] = warped[moved]
    return out


if __name__ == "__main__":
    N_FRAMES = 100
    FPS = 20
    frames = []
    for i in range(N_FRAMES):
        t = i / N_FRAMES
        frame = render(t)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)

    # Build a shared adaptive palette from the first frame so all frames use
    # the same colour table (better GIF compression than per-frame palettes).
    base = Image.fromarray(frames[0]).convert("P", palette=Image.ADAPTIVE, colors=256)

    optimized = []
    for fr in frames:
        im = Image.fromarray(fr).convert("RGB")
        im = im.quantize(palette=base, dither=Image.Dither.NONE)  # no dither: deterministic mapping
        optimized.append(im)

    # disposal=1: do not clear between frames, so unchanged pixels carry over
    # and LZW compression exploits frame-to-frame similarity.
    optimized[0].save(
        DST,
        save_all=True,
        append_images=optimized[1:],
        duration=int(1000 / FPS),
        loop=0,
        optimize=True,
        disposal=1,
    )
    print(f"Wrote {N_FRAMES} frames ({FPS} fps) to {DST}")
