# Reading list

What I would read to understand this project, in order, and where each paper shows up here.
The project's main references, with links: [how-it-works.md](how-it-works.md#references).

## Start here

1. [LangGap](https://arxiv.org/abs/2603.00592) (Hou & Zhao, IROS 2026). The introduction and the
   baseline table: most LIBERO layouts come with one task, so a model can ignore the words, and
   SmolVLA mostly fails when only the instruction changes. My test does this in the LIBERO-10 kitchen
   ([README, section 1](../README.md#1-the-test-same-kitchen-one-word-changed)).
2. [SmolVLA](https://arxiv.org/abs/2506.01844) (Shukor et al., 2025). Figure 1, the model section
   and the chunk-size ablations. Every run fine-tunes `smolvla_libero` ([train/train.sh](../train/train.sh)).
3. [LIBERO](https://arxiv.org/abs/2306.03310) (Liu et al., NeurIPS 2023). Figure 1 and how the
   suites and their 50 teleoperated demos per task were made ([the twin task](../README.md#1-the-test-same-kitchen-one-word-changed), run T's demos).
4. [Robot Learning: A Tutorial](https://arxiv.org/abs/2510.12403) (Capuano et al., 2025). Sections 4
   (ACT, action chunking) and 5 (π0, SmolVLA), with LeRobot code; LeRobot runs all my training and
   evaluation ([docs/training.md](training.md)).
5. [MimicGen](https://arxiv.org/abs/2310.17596) (Mandlekar et al., CoRL 2023). The overview figure
   and the method: re-fit object-centred segments of a demo to new layouts, execute, keep the
   successes. [sim/make_demos.py](../sim/make_demos.py) does this with my phone paths.

## How SmolVLA makes actions

- [π0](https://arxiv.org/abs/2410.24164) (Black et al., RSS 2025). The model section and Figure 3:
  a vision-language model plus an action expert trained with flow matching on [50-action chunks](../README.md#glossary).
  SmolVLA copies this design.
- [Flow Matching Guide and Code](https://arxiv.org/abs/2412.06264) (Lipman et al., 2024). A
  straight path from noise to the actions, followed in small steps at test time; `num_steps=10`
  ([train/eval.sh](../train/eval.sh)) is the number of steps.
- [ACT](https://arxiv.org/abs/2304.13705) (Zhao et al., RSS 2023). Action chunking and the
  chunk-size ablation: background for running 10 of the 50 planned actions before the model looks
  again (50 scored 0; [README, what worked](../README.md#what-worked-and-what-didnt)).
- [OpenVLA](https://arxiv.org/abs/2406.09246) (Kim et al., CoRL 2024). Appendix E, the LIBERO
  fine-tuning table. `smolvla_libero`'s data derives from OpenVLA's regenerated LIBERO demos
  (failed replays dropped): hence 41 moka-pot demos, not 50 ([train/make_moka_subset.py](../train/make_moka_subset.py)).

## Why the base model fails

- [LIBERO-Plus](https://arxiv.org/abs/2510.13626) (Fei et al., 2025). The language
  experiments: rewriting or blanking the instruction barely changes success.
  [LIBERO-PRO](https://arxiv.org/abs/2510.03827) (Zhou et al., 2025) reports the same.
- [Causal Confusion in Imitation Learning](https://arxiv.org/abs/1905.11979) (de Haan et al.,
  NeurIPS 2019). Imitation can latch onto a cue that predicts the action (the scene) instead of
  the one that causes it (the instruction).
- [Shortcut learning](https://arxiv.org/abs/2004.07780) (Geirhos et al., Nature Machine
  Intelligence 2020). Figure 1 and the part on catching shortcuts with out-of-distribution tests;
  my twin task is one ([README, design choices](../README.md#design-choices)).

## From phone video to robot data

- [Real2Render2Real](https://arxiv.org/abs/2505.09601) (Yu et al., CoRL 2025). The trajectory
  extraction: an object's 6-DoF path from one human video. Closest to my idea, but without
  physics ([how-it-works, where the ideas come from](how-it-works.md#where-the-ideas-come-from)).
- [DemoGen](https://arxiv.org/abs/2502.16932) (Xue et al., RSS 2025). Sections IV.B-C: copy the
  object part, plan the free-space part. My knob turn and grasp are scripted, the carry copied
  ([sim/kitchen_controller.py](../sim/kitchen_controller.py)).
- [X-Sim](https://arxiv.org/abs/2505.07096) (Dan et al., CoRL 2025) and
  [ORION](https://arxiv.org/abs/2405.20321) (Zhu et al., Autonomous Robots 2026). The real-to-sim
  and plan-from-video parts: both [track the object, not the hand](../README.md#design-choices).
- [Learning by Cheating](https://arxiv.org/abs/1912.12294) (Chen et al., CoRL 2019). A teacher
  that sees the true state trains an image-only student. My controller reads the simulated pan's
  pose; SmolVLA sees only images.

## Teleop demos vs phone demos

- [robomimic](https://arxiv.org/abs/2108.03298) (Mandlekar et al., CoRL 2021). Table 2: who made
  the demos, and how well, changes what the policy learns ([README, section 3](../README.md#3-the-comparison)).
- [Data Quality in Imitation Learning](https://arxiv.org/abs/2306.02437) (Belkhale et al.,
  NeurIPS 2023). Action divergence and transition diversity. My scripted demos are consistent (P
  has the lowest training loss, 0.058) but have [few tight layouts](../README.md#4-results).
- [DART](https://arxiv.org/abs/1703.09327) (Laskey et al., CoRL 2017). Clean demos show no
  recoveries; noise during collection helps. Mine are kept only [within 2 cm of the path](how-it-works.md#7-many-layouts-keep-the-successes-the-mimicgen-idea); I have
  not tried a noisier carry.

## Why mix in the moka-pot demos, and 50 vs 153

- [Counterfactually-augmented data](https://arxiv.org/abs/1909.12434) (Kaushik et al., ICLR 2020).
  Pairs that differ in one detail and in the answer teach what matters, like [one scene with two
  instructions](../README.md#design-choices) (text, so an analogy).
- LangGap again: the data-augmentation experiments, where mixing the easy LIBERO tasks back in
  lowered success. Mix ratios matter.
- [Data Scaling Laws in Imitation Learning](https://arxiv.org/abs/2410.18647) (Lin et al., ICLR
  2025). The scaling plots: more environments and objects beat more demos per environment; P and
  Pall have one kitchen, one pan.

Pall vs P is not a clean count test: Pall makes about 12 passes over its data instead of 26, and
its moka-pot demos are 21 % of its episodes instead of 45 % ([README, section 3](../README.md#3-the-comparison)).

## Tracking the pan

- [Zhang's calibration](https://doi.org/10.1109/34.888718) (Zhang, IEEE TPAMI 2000). Calibration
  from a flat board, and the board poses that add nothing. [scripts/calibrate_camera.py](../scripts/calibrate_camera.py)
  uses OpenCV's version (0.27 px).
- [ArUco](https://doi.org/10.1016/j.patcog.2014.01.005) (Garrido-Jurado et al., Pattern
  Recognition 2014) and the [OpenCV ChArUco tutorial](https://docs.opencv.org/4.x/df/d4a/tutorial_charuco_detection.html):
  marker detection, and how marker ids label the chessboard corners.
- IPPE (Collins & Bartoli, *Infinitesimal Plane-Based Pose Estimation*, IJCV 2014). Why a flat
  square has two plausible poses ([scripts/track_pan.py](../scripts/track_pan.py) keeps the one closest
  to the board's "up").
- ECC (Evangelidis & Psarakis, *Parametric Image Alignment Using Enhanced Correlation Coefficient
  Maximization*, IEEE TPAMI 2008). The template alignment that cut jitter at rest from about 5 mm
  to under 1 mm ([how-it-works, step 3](how-it-works.md#3-the-pans-position-and-rotation-in-every-frame)).

The fit onto LIBERO's kitchen ([sim/real_path.py](../sim/real_path.py)) is an exact two-point
similarity, not a least-squares fit.

## Robot side and evaluation

- [robosuite](https://arxiv.org/abs/2009.12293) (Zhu et al., 2020). Operational-space control: the
  7 numbers a demo or SmolVLA sends (hand move, hand turn, gripper) become joint torques
  ([sim/kitchen_controller.py](../sim/kitchen_controller.py)).
- [Robot Learning as an Empirical Science](https://arxiv.org/abs/2409.09491) (Kress-Gazit et al.,
  2024). Fixed start conditions, statistical tests, describing failures: what
  [train/summarize_evals.py](../train/summarize_evals.py) (Wilson intervals, Fisher tests) and
  [README, section 4](../README.md#4-results) follow.
