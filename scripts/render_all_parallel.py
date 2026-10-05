import os, sys, json
os.environ.setdefault("MUJOCO_GL", "osmesa"); os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
from multiprocessing import Pool
SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"]
OUT = "renders"

def job(args):
    name, i = args
    import imageio.v2 as imageio
    import render_scenes
    from libero.libero import benchmark
    suite = benchmark.get_benchmark_dict()[name]()
    stem = os.path.join(OUT, name, f"{i:02d}")
    if os.path.exists(stem + "_front.jpg"):
        t = suite.get_task(i); return dict(suite=name, id=i, name=t.name, language=t.language)
    task, front, wrist = render_scenes.render_task(suite, i, 256, 5)
    imageio.imwrite(stem + "_front.jpg", front, quality=90)
    imageio.imwrite(stem + "_wrist.jpg", wrist, quality=90)
    print(f"[{name} {i:02d}] {task.language}", flush=True)
    return dict(suite=name, id=i, name=task.name, language=task.language)

if __name__ == "__main__":
    from libero.libero import benchmark
    bd = benchmark.get_benchmark_dict()
    jobs = []
    for n in SUITES:
        os.makedirs(os.path.join(OUT, n), exist_ok=True)
        jobs += [(n, i) for i in range(bd[n]().n_tasks)]
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 8) as p:
        res = p.map(job, jobs, chunksize=1)
    json.dump(res, open(os.path.join(OUT, "index.json"), "w"), indent=1)
    print("DONE", len(res))
