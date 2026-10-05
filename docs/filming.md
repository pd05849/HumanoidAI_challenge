# Filming the phone clips

Goal: 15-20 short clips of a real frying pan being moved by its handle from the counter onto a
burner. Only the pan's motion is used (tracked from a printed marker); the robot does its own
grasp, and the stove knob is handled separately in simulation.

## Print

`assets/markers_letter.pdf` at **100% / actual size** (US Letter). Check the 50 mm scale bar with a
ruler.

- Page 1, ChArUco board: tape it flat on the counter next to the stove, fully visible in every
  clip. Do not move it during the session.
- Page 2, marker id 40 (80 mm): tape it flat on the inside bottom of the **cold** pan.
  Marker id 41 (40 mm) is a spare.

## Scene

- Stove **off and cold** for the whole session.
- Clear the cooktop and the path: salt, bottles, spatula and containers out of the way.
- Start position: the pan on the counter to the side of the cooktop, about 40-50 cm from the
  target burner (in LIBERO the pan starts ~47 cm from the stove).
- Target: one burner, the same for every clip.

## Camera

- Phone fixed on a stand or propped, in front of the counter, raised and looking down at about
  45 degrees, landscape. Frame the board, the pan's start spot and the target burner.
- Lock exposure and focus (tap and hold on iPhone / Android camera).
- 1080p at 30 or 60 fps.
- Before the first clip: one 10-20 s calibration clip moving the phone slowly around the board
  from different angles, then put the phone back on its stand.

## Each clip

1. Hold still for 1 s (pan on the counter).
2. Pick the pan up by the end of the handle, keeping the marker in view.
3. Lift ~10-15 cm, carry it over, lower it onto the burner, let go.
4. Hold still for 1 s.
5. Move the pan back, shifting its start a few cm each time (vary within about +-5 cm).

Keep faces out of frame. Name files in order (`clip_01.mp4`, `clip_02.mp4`, ...).
