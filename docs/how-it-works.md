# How a phone clip becomes a robot demo

The model can only learn from **robot demonstrations**: at every moment, what the robot's cameras
saw and the 7 numbers sent to the arm. A phone video of my hand has neither. So the video is used
only for **what happened to the object** (where the pan started, how it moved, where it ended
up), and the simulator produces **how the robot does it** and **what the robot sees**.

This is an *object-centric real-to-sim* approach: track the object, not the hand.

## The steps

### 1. Measure the pan's path with printed markers
- A printed **board** (ChArUco) taped to the counter defines a fixed reference frame, and is also
  filmed once to calibrate the phone camera.
- A printed **marker** inside the pan gives the pan's position and orientation in every frame.
- With OpenCV, each frame gives the camera's pose relative to the board and the pan's pose
  relative to the camera, so together **the pan's path relative to the counter**, in centimetres.
- Gaps (marker briefly hidden) are filled in and the path is smoothed.

### 2. Cut the path into phases
From the pan's motion alone: at rest on the counter, **lifted** (it starts rising), **carried**,
**put down** (it stops on the burner). No hand tracking is needed: the hand's shape doesn't
matter, only what the pan did.

### 3. Map it into the simulator
The real path is mapped onto LIBERO's kitchen so that the real start lines up with where the
simulated pan starts, and the real burner with the simulated burner. The shape of the motion
(lift height, arc, put-down) is kept.

### 4. Let the simulated robot re-enact it
In LIBERO, the Panda robot arm:
1. turns the stove knob (a fixed, pre-programmed motion, because the knob step is identical in
   both instructions and is not what's being taught);
2. grasps the pan by its handle (a fixed grasp relative to the pan, the same for every demo);
3. moves so that the **simulated pan follows my real pan's path**;
4. lets go on the burner.

The physics engine decides whether it worked, and LIBERO's own goal check ("stove on and pan on
the burner") labels success. While this runs, the simulator records the robot's camera images
and the arm commands: **one robot demo, created from one phone clip.**

#### How the arm's motion is computed

The robot's action at each step is **7 numbers**: a small hand translation (x, y, z), a small hand
rotation (axis-angle), and the gripper (-1 open, +1 closed). LIBERO's built-in operational-space
controller turns these into torques for the arm's 7 joints, so no joint angles are computed here.

1. **Pan path → hand path.** While the pan is held, the gripper and pan move as one rigid
   piece. So the hand pose the robot needs is the pan pose from the video composed with the fixed
   grasp:
   `T_world_hand(t) = T_world_pan(t) · T_pan_grasp`.
2. **Segments.** The motion is split as in DemoGen: *free-space* parts (reach the knob, reach the
   handle, retreat) are re-planned as smooth straight-line moves to the next segment's start;
   the *object* part (lift, carry, put down) is copied from the video, moved along with the
   objects.
3. **Fitting a new layout** (MimicGen idea). When the pan or stove start elsewhere, the carry
   path is bent so it starts at the new pan pose and ends at the new burner, blending gradually
   from one correction to the other along the path; rotations are blended the same way.
4. **Timing.** The phone path (30-60 frames/s) is resampled to the robot's 20 steps/s, with a
   speed limit similar to LIBERO's human demos.
5. **Tracking.** At every step a proportional controller converts "where the hand should be"
   minus "where it is" into the 7 action numbers (clipped to LIBERO's limits). The physics then
   runs, so the recorded actions are exactly what the robot really did.

This differs from Real2Render2Real, which also starts from phone captures but computes robot
motion with kinematics only and renders it, without a physics engine. Here every generated demo
is executed in LIBERO's physics and kept only if LIBERO's goal check passes, and the images are
LIBERO's own renders, the same kind the model sees at test time.

### 5. Multiply
The same clip is replayed from many different starting layouts (the objects shift slightly
between tries). Because the motion is stored *relative to the pan and the stove*, it adapts to
where they are. Only successful replays are kept. This is the idea behind **MimicGen**, written
from scratch here because the official package doesn't install alongside LeRobot's LIBERO.
About 15 clips become about 200 robot demos.

### 6. Fine-tune and test
The demos are saved in LeRobot's dataset format and used to fine-tune `smolvla_libero`, mixed with
the 50 original moka-pot demos so the model must use the words to tell the two tasks apart. A
second model is fine-tuned the same way on the 50 human teleoperation demos of the twin task, so
the two can be compared fairly.

## What is honest to say about this
- The grasp and the knob turn are not learned from my video; they are fixed motions. The video
  contributes the **task** (which object, where to, along which path), which is exactly the part
  the zero-shot model gets wrong.
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
