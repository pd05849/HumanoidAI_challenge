Left: one of the LIBERO teleop demos (a person steering the robot). Right: one phone-derived demo (knob turn, grasp and release scripted; the carry follows the pan path tracked in a phone video). Same task, same simulator, the two cameras the policy sees (mirrored left-right, as LIBERO stores them), 20 control steps per second. The bar under each side shows when the gripper is closed; the black tick is now.

Median over each dataset's demos, (min–max) in brackets. Times count 20 Hz control steps. No-op steps (arm still, gripper unchanged) are not in the datasets: the phone demos have almost none, but LIBERO's converter skips the operator's pauses, so teleop times are active time. "Pan in hand" is the longest closed-gripper stretch with the fingers stopped on something; a regrasp splits it.

| median (min–max) | LIBERO teleop, 50 demos | phone-derived, 50 demos |
|---|---|---|
| demo length (s) | 13.7 (11.9–16.8) | 13.8 (12.8–15.7) |
| gripper travel, whole demo (m) | 1.85 (1.63–2.01) | 1.79 (1.66–1.97) |
| pan in hand (s) | 3.8 (3.3–5.5) | 6.8 (6.0–8.7) |
| pan moved, grasp to release, straight line (cm) | 44 (39–49) | 40 (37–43) |
| pan lifted: highest point above the grasp (cm) | 21 (16–30) | 24 (20–29) |
| gripper speed while moving, mean (cm/s) | 14.2 (12.4–16.0) | 13.7 (12.5–15.0) |
| gripper speed, fastest 5 % of steps (cm/s) | 25 (21–27) | 41 (34–43) |
| translation command change per step, mean (lower = smoother) | 0.030 (0.023–0.044) | 0.034 (0.031–0.039) |
| largest translation command (limit 1; LIBERO's 3-D mouse reaches 0.94) | 0.94 (0.83–0.94) | 1.00 (1.00–1.00) |
| steps at full translation speed (command ≥ 0.93) | 1% (0%–7%) | 14% (11%–18%) |
| steps with the gripper commanded closed | 45% (34%–51%) | 84% (82%–86%) |
| closed-gripper stretches per demo (knob, carry, regrasps) | 2 (2–4) | 2 (2–2) |
