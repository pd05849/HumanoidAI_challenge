"""Run `lerobot-eval` unchanged, but also show the robot's front camera live in a window.

Usage: replace `lerobot-eval` with `python watch_eval.py` and keep every flag, e.g.
    MUJOCO_GL=egl python watch_eval.py --policy.path=HuggingFaceVLA/smolvla_libero \
        --env.type=libero --env.task=libero_90 --env.task_ids='[21]' \
        --env.observation_height=256 --env.observation_width=256 \
        --eval.n_episodes=20 --eval.batch_size=1 --output_dir=./eval_logs/pilot_pan
Needs --eval.batch_size=1 (the simulator then runs inside this process).
"""
import os
import sys

import numpy as np

import lerobot.envs.libero as libero_env

SHOW_EVERY = 2  # draw every 2nd control step (the sim runs at 20 steps/s)
SCALE = 2       # enlarge the 256 px camera image


class Viewer:
    """OpenCV window if available, else a matplotlib window, else a live.jpg file."""

    def __init__(self):
        self.mode = "file"
        has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
        if not has_display:  # GUI toolkits abort (not just raise) without a display
            self._file_mode()
            return
        try:
            import cv2
            cv2.namedWindow("LIBERO live", cv2.WINDOW_NORMAL)  # fails on opencv-headless builds
            self.cv2, self.mode = cv2, "cv2"
            return
        except Exception:
            pass
        try:
            import matplotlib
            matplotlib.use("TkAgg")
            import matplotlib.pyplot as plt
            plt.ion()
            self.plt = plt
            self.fig, self.ax = plt.subplots(figsize=(6, 6.4))
            self.ax.axis("off")
            self.im = None
            self.mode = "mpl"
            return
        except Exception:
            pass
        self._file_mode()

    def _file_mode(self):
        import imageio.v2 as imageio
        self.imageio, self.mode = imageio, "file"
        print("[watch_eval] no GUI window available -> updating live.jpg (open it in an image viewer)")

    def show(self, img, title):
        img = np.ascontiguousarray(img)
        if self.mode == "cv2":
            big = self.cv2.resize(img, None, fx=SCALE, fy=SCALE, interpolation=self.cv2.INTER_NEAREST)
            big = self.cv2.cvtColor(big, self.cv2.COLOR_RGB2BGR)
            self.cv2.putText(big, title, (8, 22), self.cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            self.cv2.imshow("LIBERO live", big)
            self.cv2.waitKey(1)
        elif self.mode == "mpl":
            if self.im is None:
                self.im = self.ax.imshow(img)
            else:
                self.im.set_data(img)
            self.ax.set_title(title, fontsize=9)
            self.plt.pause(0.001)
        else:
            self.imageio.imwrite("live.jpg", img)


_viewer = None
_step_count = 0
_orig_step = libero_env.LiberoEnv.step
_orig_reset = libero_env.LiberoEnv.reset


def _step(self, action):
    global _viewer, _step_count
    out = _orig_step(self, action)
    _step_count += 1
    if _step_count % SHOW_EVERY == 0:
        if _viewer is None:
            _viewer = Viewer()
        _viewer.show(self.render(), f"{self.task_description}  |  step {_step_count}")
    return out


def _reset(self, *args, **kwargs):
    global _step_count
    _step_count = 0
    return _orig_reset(self, *args, **kwargs)


libero_env.LiberoEnv.step = _step
libero_env.LiberoEnv.reset = _reset

from lerobot.scripts.lerobot_eval import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
