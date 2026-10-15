---
license: apache-2.0
task_categories:
- robotics
tags:
- LeRobot
- LIBERO
- libero_90
- frying-pan
- real-to-sim
- phone-video
- smolvla
- panda
configs:
- config_name: default
  data_files: data/*/*.parquet
---

This dataset was created using [LeRobot](https://github.com/huggingface/lerobot).


<a class="flex" href="https://huggingface.co/spaces/lerobot/visualize_dataset?path=Delcastillo8/libero90_frying_pan_phone">
<img class="block dark:hidden" src="https://huggingface.co/datasets/huggingface/badges/resolve/main/visualize-this-dataset-xl.svg"/>
<img class="hidden dark:block" src="https://huggingface.co/datasets/huggingface/badges/resolve/main/visualize-this-dataset-xl-dark.svg"/>
</a>


## Dataset Description

Robot demonstrations of LIBERO-90 task 21, **"turn on the stove and put the frying pan on it"**,
made from phone videos of a person moving a real frying pan in a real kitchen.

How each demo was made ([code and write-up](https://github.com/pd05849/HumanoidAI_challenge)):
1. A printed marker inside a real pan and a printed board on the stove were filmed with an iPad
   (18 takes, 17 of which gave demos). The pan's 3-D path was tracked in every frame.
2. Each path was fitted onto LIBERO's kitchen (rotated, scaled, shifted onto the simulated pan
   and burner; played 1.5x faster; raised locally to clear the moka pot).
3. A scripted Franka Panda in LIBERO turned the stove knob, grasped the pan's handle and carried
   the pan along the path, with physics; only episodes that passed LIBERO's own success check
   and quality gates (path followed within 2 cm RMS, moka pot undisturbed) were kept.
4. Recorded in the conventions of `HuggingFaceVLA/libero` (the data `smolvla_libero` was trained
   on): `observation.images.image` (front) and `observation.images.image2` (wrist), 256x256,
   rotated 180 degrees as LeRobot's LIBERO processor does; `observation.state` 8-D (end-effector
   position, axis-angle, gripper joints); `action` 7-D; fps label 10, one frame per 20 Hz
   control step. Frames are stored as AV1 video (`HuggingFaceVLA/libero` stores images in the
   parquet files; LeRobot reads both).

153 episodes, 42407 frames. The knob turn and the grasp are scripted; the carry
(where the pan goes, how high, how it turns, how fast) comes from the phone videos. None of the
episodes starts from LIBERO's 50 fixed evaluation layouts (nearest: 0.3-2.5 cm).

All 153 demos, used to train run Pall
([`Delcastillo8/smolvla_libero_frying_pan_Pall`](https://huggingface.co/Delcastillo8/smolvla_libero_frying_pan_Pall)).
The 50 demos of run P are in [`Delcastillo8/libero90_frying_pan_phone_50`](https://huggingface.co/datasets/Delcastillo8/libero90_frying_pan_phone_50).


- **Homepage:** https://github.com/pd05849/HumanoidAI_challenge
- **License:** apache-2.0

## Dataset Structure

[meta/info.json](https://huggingface.co/datasets/Delcastillo8/libero90_frying_pan_phone/blob/main/meta/info.json):
```json
{
    "codebase_version": "v3.0",
    "fps": 10,
    "features": {
        "observation.images.image": {
            "dtype": "video",
            "shape": [
                256,
                256,
                3
            ],
            "names": [
                "height",
                "width",
                "channels"
            ],
            "info": {
                "video.height": 256,
                "video.width": 256,
                "video.codec": "av1",
                "video.pix_fmt": "yuv420p",
                "video.fps": 10,
                "video.channels": 3,
                "has_audio": false,
                "video.g": 2,
                "video.crf": 30,
                "video.preset": 12,
                "video.fast_decode": 0,
                "video.video_backend": "pyav",
                "video.extra_options": {},
                "is_depth_map": false
            }
        },
        "observation.images.image2": {
            "dtype": "video",
            "shape": [
                256,
                256,
                3
            ],
            "names": [
                "height",
                "width",
                "channels"
            ],
            "info": {
                "video.height": 256,
                "video.width": 256,
                "video.codec": "av1",
                "video.pix_fmt": "yuv420p",
                "video.fps": 10,
                "video.channels": 3,
                "has_audio": false,
                "video.g": 2,
                "video.crf": 30,
                "video.preset": 12,
                "video.fast_decode": 0,
                "video.video_backend": "pyav",
                "video.extra_options": {},
                "is_depth_map": false
            }
        },
        "observation.state": {
            "dtype": "float32",
            "shape": [
                8
            ],
            "names": null
        },
        "action": {
            "dtype": "float32",
            "shape": [
                7
            ],
            "names": null
        },
        "timestamp": {
            "dtype": "float32",
            "shape": [
                1
            ],
            "names": null
        },
        "frame_index": {
            "dtype": "int64",
            "shape": [
                1
            ],
            "names": null
        },
        "episode_index": {
            "dtype": "int64",
            "shape": [
                1
            ],
            "names": null
        },
        "index": {
            "dtype": "int64",
            "shape": [
                1
            ],
            "names": null
        },
        "task_index": {
            "dtype": "int64",
            "shape": [
                1
            ],
            "names": null
        }
    },
    "total_episodes": 153,
    "total_frames": 42407,
    "total_tasks": 1,
    "chunks_size": 1000,
    "data_files_size_in_mb": 90,
    "video_files_size_in_mb": 90,
    "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
    "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
    "robot_type": "panda",
    "splits": {
        "train": "0:153"
    }
}
```
