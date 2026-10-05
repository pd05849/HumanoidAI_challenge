"""Build captioned contact sheets from renders/index.json (one per suite; LIBERO-90 split in 3)."""
import json
import os
import textwrap

from PIL import Image, ImageDraw, ImageFont

R = "renders"
idx = json.load(open(os.path.join(R, "index.json")))
try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    big = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
except OSError:
    font = big = ImageFont.load_default()

TILE, CAP, COLS = 256, 52, 5
TITLES = {"libero_spatial": "LIBERO-Spatial", "libero_object": "LIBERO-Object", "libero_goal": "LIBERO-Goal",
          "libero_10": "LIBERO-10 (Long)", "libero_90": "LIBERO-90"}


def sheet(items, title, path):
    rows = (len(items) + COLS - 1) // COLS
    W, H = COLS * TILE, 40 + rows * (TILE + CAP)
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    d.text((10, 8), title, fill="black", font=big)
    for k, it in enumerate(items):
        x, y = (k % COLS) * TILE, 40 + (k // COLS) * (TILE + CAP)
        img.paste(Image.open(os.path.join(R, it["suite"], f"{it['id']:02d}_front.jpg")).resize((TILE, TILE)), (x, y))
        cap = textwrap.wrap(f"{it['id']}: {it['language']}", 36)[:3]
        for j, line in enumerate(cap):
            d.text((x + 4, y + TILE + 2 + 16 * j), line, fill="black", font=font)
    img.save(path, quality=85)
    print(path, img.size)


os.makedirs("sheets", exist_ok=True)
for s in TITLES:
    items = [i for i in idx if i["suite"] == s]
    if s == "libero_90":
        for p in range(3):
            sheet(items[p * 30:(p + 1) * 30], f"{TITLES[s]} (tasks {p*30}-{p*30+29})", f"sheets/{s}_{p+1}.jpg")
    else:
        sheet(items, TITLES[s], f"sheets/{s}.jpg")
