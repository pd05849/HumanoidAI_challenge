"""Check that this Python environment can run the project: LeRobot, SmolVLA, LIBERO, the simulator's
rendering, and the GPU.

    MUJOCO_GL=egl python scripts/check_env.py        # NVIDIA machine (training / evaluation)
    MUJOCO_GL=osmesa python scripts/check_env.py     # no NVIDIA GPU (needs: sudo apt install libosmesa6)

Prints one line per check and, if something is missing, the command that installs it.
"""
import importlib
import importlib.metadata as md
import os
import sys

ok = True


def check(name, fn):
    global ok
    try:
        msg = fn()
        print(f"  OK       {name}{': ' + msg if msg else ''}")
    except Exception as ex:  # noqa: BLE001 - report every failure the same way
        ok = False
        print(f"  MISSING  {name}: {type(ex).__name__}: {str(ex).splitlines()[0][:120] if str(ex) else ''}")


def version(pkg):
    return lambda: md.version(pkg)


def imports(mod):
    return lambda: (importlib.import_module(mod), "")[1]


def gpu():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("torch sees no CUDA GPU (fine for tracking and demo generation; "
                           "training and evaluation need one)")
    p = torch.cuda.get_device_properties(0)
    return f"{p.name}, {p.total_memory / 1e9:.0f} GB, torch {torch.__version__}"


def render():
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    task = benchmark.get_benchmark_dict()["libero_90"]().get_task(21)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=128, camera_widths=128)
    env.seed(0)
    obs = env.reset()
    env.close()
    return f"'{task.language}' rendered {obs['agentview_image'].shape} with MUJOCO_GL={os.environ.get('MUJOCO_GL')}"


print(f"Python {sys.version.split()[0]} at {sys.executable}")
print("LeRobot and SmolVLA")
check("lerobot", version("lerobot"))
check("SmolVLA policy", imports("lerobot.policies.smolvla.modeling_smolvla"))
check("transformers", version("transformers"))
check("num2words (SmolVLA's text processor)", version("num2words"))
check("accelerate (needed by lerobot-train)", version("accelerate"))
print("LIBERO simulator")
check("hf-libero", version("hf-libero"))
check("mujoco", version("mujoco"))
check("LeRobot's LIBERO env", imports("lerobot.envs.libero"))
check("LIBERO-90 task 21 renders", render)
print("GPU")
try:  # informational: tracking, demo generation and the analysis run on the CPU
    print(f"  OK       CUDA: {gpu()}")
except Exception as ex:  # noqa: BLE001
    print(f"  NO GPU   {ex}")

if not ok:
    print('\nFix missing packages with:  uv pip install "lerobot[libero,smolvla,training]==0.6.1"'
          "  (needs Python >= 3.12)"
          "\nNo Python 3.12 here? uv makes one without touching the system Python:"
          "\n  curl -LsSf https://astral.sh/uv/install.sh | sh && source $HOME/.local/bin/env"
          "\n  uv venv --python 3.12 .venv && source .venv/bin/activate"
          '\n  uv pip install "lerobot[libero,smolvla,training]==0.6.1"'
          "\nIf only rendering fails: on an NVIDIA machine set MUJOCO_GL=egl; without one, "
          "sudo apt install libosmesa6 and set MUJOCO_GL=osmesa.")
sys.exit(0 if ok else 1)
