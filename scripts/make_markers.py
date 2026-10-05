"""Printable marker sheets (US Letter, print at 100% / "actual size").

Page 1: ChArUco board (5x7 squares, 30 mm squares, 22 mm markers, DICT_4X4_50 ids 0-16).
        Tape it flat on the counter next to the stove: it defines the table frame, and the same
        board is used once to calibrate the phone camera.
Page 2: pan markers (DICT_4X4_50): id 40 at 80 mm for the inside of the pan, id 41 at 40 mm
        as a spare / handle option.
Each page has a 50 mm scale bar to check the printer did not rescale.
"""
import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

PAGE_W, PAGE_H = 215.9, 279.4  # US Letter in mm
PX_PER_MM = 10
DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)


def place(fig, img, x_mm, y_mm, w_mm, h_mm):
    """Draw img with its top-left corner at (x_mm, y_mm) from the page's top-left, at exact size."""
    ax = fig.add_axes([x_mm / PAGE_W, 1 - (y_mm + h_mm) / PAGE_H, w_mm / PAGE_W, h_mm / PAGE_H])
    ax.imshow(img, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    ax.axis("off")


def scale_bar(fig, x_mm, y_mm):
    ax = fig.add_axes([0, 0, 1, 1], facecolor="none")
    ax.set_xlim(0, PAGE_W)
    ax.set_ylim(PAGE_H, 0)
    ax.axis("off")
    ax.plot([x_mm, x_mm + 50], [y_mm, y_mm], color="black", lw=2)
    for t in (x_mm, x_mm + 50):
        ax.plot([t, t], [y_mm - 2, y_mm + 2], color="black", lw=2)
    ax.text(x_mm + 55, y_mm + 1.2, "50 mm: measure this; if it is not 50 mm, reprint at 100% / actual size",
            fontsize=7, va="center")
    return ax


def text(fig, x_mm, y_mm, s, size=9, bold=False):
    fig.text(x_mm / PAGE_W, 1 - y_mm / PAGE_H, s, fontsize=size, va="top",
             fontweight="bold" if bold else "normal")


with PdfPages("markers_letter.pdf") as pdf:
    # page 1: ChArUco board
    sq, mk, nx, ny = 30.0, 22.0, 5, 7
    board = cv2.aruco.CharucoBoard((nx, ny), sq / 1000, mk / 1000, DICT)
    bw, bh = nx * sq, ny * sq
    img = board.generateImage((int(bw * PX_PER_MM), int(bh * PX_PER_MM)), marginSize=0, borderBits=1)
    fig = plt.figure(figsize=(PAGE_W / 25.4, PAGE_H / 25.4))
    text(fig, 15, 12, "Page 1 - counter board (ChArUco 5x7, 30 mm squares, 22 mm markers, DICT_4X4_50)", 10, True)
    text(fig, 15, 19, "Tape flat on the counter next to the stove, fully visible in every clip. Do not move it during a session.")
    text(fig, 15, 24, "Also film it once from many angles (10-20 s, slow) to calibrate the phone camera.")
    place(fig, img, (PAGE_W - bw) / 2, 34, bw, bh)
    scale_bar(fig, 20, 34 + bh + 12)
    pdf.savefig(fig)
    fig.savefig("markers_page1.png", dpi=60)
    plt.close(fig)

    # page 2: pan markers
    fig = plt.figure(figsize=(PAGE_W / 25.4, PAGE_H / 25.4))
    text(fig, 15, 12, "Page 2 - pan markers (DICT_4X4_50)", 10, True)
    text(fig, 15, 19, "id 40 (80 mm): tape flat on the inside bottom of the COLD pan, so the camera sees it all the way.")
    text(fig, 15, 24, "id 41 (40 mm): spare, or on top of the handle near the pan if id 40 gets hidden. Keep the white border.")
    for mid, size, x, y in [(40, 80.0, 25, 40), (41, 40.0, 130, 60)]:
        border = 8.0  # white quiet zone around the marker, then a thin cut line
        m = cv2.aruco.generateImageMarker(DICT, mid, int(size * PX_PER_MM))
        canvas = 255 * __import__("numpy").ones((int((size + 2 * border) * PX_PER_MM),) * 2, dtype="uint8")
        o = int(border * PX_PER_MM)
        canvas[o:o + m.shape[0], o:o + m.shape[1]] = m
        canvas[0, :] = canvas[-1, :] = canvas[:, 0] = canvas[:, -1] = 160  # cut line
        place(fig, canvas, x, y, size + 2 * border, size + 2 * border)
        text(fig, x, y + size + 2 * border + 3, f"id {mid}  ({int(size)} mm)", 8)
    scale_bar(fig, 20, 160)
    pdf.savefig(fig)
    fig.savefig("markers_page2.png", dpi=60)
    plt.close(fig)
print("wrote markers_letter.pdf")
