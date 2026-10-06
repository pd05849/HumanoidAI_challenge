# How a phone clip becomes a robot demo

A VLA learns from **robot demonstrations**: at every moment, what the robot's cameras saw and the 7
numbers sent to the arm. A phone video of my hand has neither. So the video is used only for
**what happened to the object** (where the pan started, how it rose, moved and turned, where it
landed), and the simulator produces **how the robot does it** and **what the robot sees**. This is
an *object-centric real-to-sim* approach: track the object, not the hand.

```
phone clip -> (1) calibrate the camera -> (2) camera pose from the board -> (3) pan pose in every frame
           -> (4) clean path + phases -> (5) fit onto LIBERO's kitchen -> (6) robot re-enacts it in physics
           -> (7) repeat over many layouts, keep the successes -> (8) save as LeRobot training data
```

## 1. Camera calibration: from pixels to centimetres

A camera maps 3-D points to pixels through a few numbers specific to its lens: the **focal
length** (how much it zooms, in pixels), the **principal point** (where the lens centre falls in
the image) and **lens distortion**. Without them pixels cannot be turned into centimetres.

The printed board is a **ChArUco** board: a chessboard (whose corners can be located to a fraction
of a pixel) with small ArUco markers in the white squares (which say *which* corner is which, so
a partly hidden or steeply tilted board still works). For calibration the board is held and moved
in front of the fixed iPad at many angles; OpenCV's `calibrateCamera` (Zhang's method) finds the
lens numbers that best explain where every corner appeared in every frame.

- Model: pinhole + radial distortion (k1, k2). Richer models fitted the same data only
  marginally better and disagreed between clips; this one gave the same answer for clips recorded
  at 30 and 60 fps (focal length within 0.2 %).
- Result for the iPad (`data/calibration/ipad_1080p.json`): focal length 1850 / 1857 px, principal
  point (946, 508), k1 0.145, k2 -0.295, reprojection error **0.27 px** over 1124 frames.
- A detail that mattered: the second copy of the board, taped on the burner, was also in view
  during calibration; both copies carry the same marker ids and confused the detector. The script
  keeps only the large markers (the held board, close to the camera) in every frame.

## 2. Where is the camera? The board defines the world

The board lying on the stove is the fixed reference: its front-left corner is the origin, **x**
points right along the counter, **y** toward the wall, **z** up. From the board's corners and the
calibration, `solvePnP` gives the camera's position and angle relative to the board (here about
0.9 m above the cooktop, looking down 34°). The iPad is on a tripod, so this is computed **once
per clip** from the median position of each corner, and each frame is corrected for the image's
~1 px drift (video stabilisation).

## 3. The pan's position and rotation in every frame

A pose is a position plus a rotation, and poses chain: *pan in world = camera in world x pan as
seen by the camera*. For each frame:

1. **Find the marker** (ArUco) inside the pan: its 4 corner pixels. A second, more permissive pass
   catches blurred frames.
2. **Refine with template matching (ECC).** Seen from a shallow angle, the pan's front rim covers
   part of the marker's white border and pulls one detected corner ~10 px off, which made the pose
   jump by up to 4 cm. ECC aligns a perfect image of the marker to the frame using **all** its
   black and white cells, so one hidden edge barely matters (jitter at rest dropped from 12 mm to
   ~0.3 mm). It also tracks through frames the detector cannot read.
3. **Corners -> 3-D pose** with `solvePnP` (IPPE for a flat square of known size, 80 mm). A flat
   square seen from afar has two near-identical 3-D solutions (like a coin tilted toward or away
   from you); the one whose "up" is closest to the board's up is kept, since the pan stays roughly
   level.
4. **Plausibility checks:** a pose outside the kitchen workspace, or more than 8 cm from the median
   of its neighbours within ±0.25 s, is rejected (one take had a false detection while the pan was
   tilted steeply toward the camera).

Result: x, y, z of the marker centre (the centre of the pan's bottom) and roll, pitch, yaw. Only
the **change in yaw** is used later (the marker was taped at an arbitrary angle to the handle, and
LIBERO's pan always starts at the same angle); tilt is measured but not replayed (the robot keeps
the pan level).

## 4. Clean path and phases

Short gaps are filled with a smooth curve and the path is lightly smoothed (Savitzky-Golay,
~0.25 s). The pan's height then splits each take into phases: **rest** (still, within a noise
threshold), **lift-off** (leaves a 3 mm band above its start height), **peak**, **touchdown**, rest.
`scripts/render_tracking.py` draws all of this back onto the video so it can be checked by eye.

Quality over the 18 takes: the pan was measured in 95-100 % of frames, jitter at rest 0.35-0.83 mm,
carries of 48-53 cm with lifts of 20-25 cm. An independent re-implementation (different corner
method and pose solver) on a test clip agreed to 0.4 cm in distance and 0.2 cm in lift.

## 5. Fitting the path onto LIBERO's kitchen

My kitchen and LIBERO's have different sizes, so the path is fitted onto the simulator, separately
for every layout of the simulated objects (`sim/real_path.py`):

- **Position (top view): a similarity transform** (rotation + one uniform scale + shift, never a
  mirror image), fixed by two pairs of points: my start -> the simulated pan's start (right after
  the robot grasps it), my end -> the place spot on the simulated burner. For my kitchen this comes
  out as a rotation of about -90° (my "toward the wall" becomes the robot's "away from me": I stand
  at the counter as the robot stands at its table, stove on the left, pan on the right) and a
  scale of about 0.8-0.9 (my ~51 cm become the simulator's ~42 cm).
- **Height:** my lift curve rides on a baseline that rises smoothly from the table to the burner
  (the simulated stove is 3.2 cm higher). A MuJoCo pre-flight check measures the clearance to the
  simulator's 15 cm moka pot (my counter has nothing there); where the pan would come within 2 cm,
  the path is raised **only in that stretch**.
- **Rotation:** the simulated start angle plus my yaw change.
- **Timing:** resampled to the robot's 20 control steps per second and sped up 1.5x (my carries
  took 7-11 s; the evaluation gives the robot 400 steps = 20 s for the whole task), with a speed
  limit of 0.25 m/s, about the speed of the robot's own scripted motions.

## 6. The robot re-enacts it, with physics

| Part | Comes from |
|---|---|
| Turn the stove knob | scripted (identical for both instructions, not what is being taught) |
| Grasp the pan's handle | scripted, with a collision check so the wide gripper does not hit the moka pot |
| **Lift, carry, turn, set down: path and timing** | **the phone video** |
| Release on the burner | scripted |

Every 1/20 s the controller (`sim/kitchen_controller.py`) looks at where the **simulated pan
actually is** and where my path says it should be 0.2 s from now (to cancel the arm's lag), and
turns the difference into the robot's 7 numbers: a small hand translation and rotation, gripper
closed. LIBERO's built-in operational-space controller turns those into joint torques and MuJoCo
simulates contact, friction and gravity, so the pan really is held by friction in the gripper.
Steering on the live pan pose also corrects for the pan slipping a little in the grasp. LIBERO's
own goal check ("stove on, pan on the burner") decides success.

Checked on a first test clip: the simulated pan followed the fitted path to **0.5 cm** (RMS), and
the task succeeded in **10/10** start layouts plus **10/10** held-out ones. An independent re-run
reproduced this exactly, and confirmed the robot follows the real path rather than a built-in one:
adding a 6 cm detour to the input path produced the same detour in the simulation.

## 7. Many layouts, keep the successes (the MimicGen idea)

`sim/make_demos.py` repeats step 5-6 for every take over **new random layouts** (LIBERO places the
pan, moka pot and stove at random within their allowed regions; these are never the 50 fixed test
layouts). Each episode is first simulated without cameras (fast); only if it succeeds is it re-run
with cameras and recorded. Quality gates reject episodes that knock the moka pot (> 10 mm or > 10°)
or where the pan swings in the hand (path error > 2 cm), and the next layout is tried instead.
About 1 in 5 layouts has no collision-free grasp (the moka pot sits next to the handle) and is
skipped.

## 8. Saved as training data

Every kept episode is recorded **exactly as the model sees and acts** at test time (LeRobot's
LIBERO conventions): the front and wrist camera images (256x256, rotated 180° like LeRobot's
processor), the 8-number robot state (hand position, hand rotation as axis-angle, gripper), the
7-number action actually sent, and the instruction "turn on the stove and put the frying pan on
it". Steps where the robot does nothing are dropped, as in LIBERO's training data.

This was checked against LeRobot's own evaluation environment: rebuilding a recorded layout there
gives **pixel-identical** images and a bit-identical state, and replaying the recorded actions
reproduces the success at the same step. (An early version of the generator failed this check:
it started the simulated arm with robosuite's joint noise, which changes the controller's posture
target; it was fixed before any data was generated.)

## Where the ideas come from

| Method | What it does | What this project takes from it | What is different here |
|---|---|---|---|
| **MimicGen** (Mandlekar 2023) | expands a few *robot teleop* demos by re-fitting object-centred segments to new object layouts, executing them in simulation and keeping the successes | motion stored relative to the objects, re-fitted per layout, **executed and filtered by success** (steps 5-7) | the source is a **human phone video**; the **object's** path is re-fitted and the controller derives the gripper motion |
| **DemoGen** (Xue 2025) | expands one demo by re-planning **free-space** motion and moving the **object-contact** part with the object; synthesises point-cloud observations | free-space parts (reach the knob, reach the handle, retreat) are planned by the controller; the object part comes from the video | **real physics** and LIBERO's own rendered images instead of edited point clouds |
| **Real2Render2Real** (Yu 2025) | phone scan + human video -> the object's 6-DoF path -> robot motion by inverse kinematics, rendered **without physics** | phone video -> **object's 6-DoF path** as the signal (steps 1-4) | printed markers instead of a 3-D scan (simpler, precise for one object); every demo is **physics-checked**; the target is an existing benchmark scene, so training and test images look alike |

In one sentence: R2R2R-style phone capture of the object's path, MimicGen-style re-fitting and
success filtering in physics, DemoGen-style split between planned free motion and copied object
motion, used to teach a model **one missing instruction word** in a same-scene twin task.

## What is honest to say about this

- The knob turn and the grasp are scripted, not learned from the video. The video contributes the
  **task** (which object, where to, along which path, how high, how fast), which is exactly the part
  the unchanged model gets wrong.
- The start and end of each path are fitted onto the simulator, so distances are scaled (about 0.8-0.9x)
  and absolute positions are not copied; tilt is not replayed; the path is raised near the
  simulator-only moka pot when needed.
- All 18 takes come from one person in one kitchen, so the paths are similar to each other.
- Everything happens in simulation; no real robot is used.

## References

**Benchmark and models**
- Liu et al., *LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning*, NeurIPS 2023.
  [arXiv:2306.03310](https://arxiv.org/abs/2306.03310) · [code](https://github.com/Lifelong-Robot-Learning/LIBERO)
- Shukor et al., *SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics*,
  2025. [arXiv:2506.01844](https://arxiv.org/abs/2506.01844)
- [LeRobot](https://github.com/huggingface/lerobot) (training/evaluation framework) and its
  [LIBERO docs](https://github.com/huggingface/lerobot/blob/main/docs/source/libero.mdx)

**Do LIBERO models really follow language?**
- Hou & Zhao, *LangGap: Diagnosing and Closing the Language Gap in Vision-Language-Action Models*,
  IROS 2026. [arXiv:2603.00592](https://arxiv.org/abs/2603.00592) · [code](https://github.com/YC11Hou/langgap)
  (same-scene instruction changes on LIBERO Spatial/Object/Goal; not the LIBERO-10 kitchen used here)
- *LIBERO-Plus*: robustness analysis of VLAs on LIBERO, finds models largely ignore language.
  [arXiv:2510.13626](https://arxiv.org/abs/2510.13626) · [code](https://github.com/sylvestf/LIBERO-plus)
- *LIBERO-PRO*: perturbation benchmark showing collapse when the instruction's target changes.
  [code](https://github.com/Zxy-MLlab/LIBERO-PRO)

**Turning human video into robot data (the approach above)**
- Mandlekar et al., *MimicGen: A Data Generation System for Scalable Robot Learning using Human
  Demonstrations*, CoRL 2023. [arXiv:2310.17596](https://arxiv.org/abs/2310.17596) ·
  [site](https://mimicgen.github.io/)
- Xue et al., *DemoGen: Synthetic Demonstration Generation for Data-Efficient Visuomotor Policy
  Learning*, RSS 2025. [arXiv:2502.16932](https://arxiv.org/abs/2502.16932) ·
  [code](https://github.com/TEA-Lab/DemoGen) (one human demo expanded into many by moving the trajectory with the objects)
- Yu et al., *Real2Render2Real: Scaling Robot Data Without Dynamics Simulation or Robot Hardware*,
  CoRL 2025. [arXiv:2505.09601](https://arxiv.org/abs/2505.09601) ·
  [code](https://github.com/uynitsuj/real2render2real) (phone object scan + one human video)
- Dan et al., *X-Sim: Cross-Embodiment Learning via Real-to-Sim-to-Real*, CoRL 2025.
  [arXiv:2505.07096](https://arxiv.org/abs/2505.07096) · [site](https://portal-cornell.github.io/X-Sim/)
  (object motion from human video as the transferable signal)
- Lum et al., *Crossing the Human-Robot Embodiment Gap with Sim-to-Real RL using One Human
  Demonstration* (Human2Sim2Robot), 2025. [arXiv:2504.12609](https://arxiv.org/abs/2504.12609)
- Bahl et al., *Affordances from Human Videos as a Versatile Representation for Robotics* (VRB),
  CVPR 2023. [arXiv:2304.08488](https://arxiv.org/abs/2304.08488)
- *H2RBench: A Real-to-Sim Benchmark for Evaluating Human-to-Robot Transfer*, CoRL 2026.
  [code](https://github.com/xiaochy/H2RBench)

**Splitting "where to go" from "how to grasp"**
- Nguyen et al., *GloVLA: Let Geometry Move and Local VLA Interact for Robust Object-Centric
  Manipulation in Unstructured Environments*, 2026. [arXiv:2609.06256](https://arxiv.org/abs/2609.06256)

**Markers and camera calibration**
- Garrido-Jurado et al., *Automatic generation and detection of highly reliable fiducial markers
  under occlusion*, Pattern Recognition 47(6), 2014 (ArUco).
- [OpenCV ArUco / ChArUco tutorial](https://docs.opencv.org/4.x/df/d4a/tutorial_charuco_detection.html)
