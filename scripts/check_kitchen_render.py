import os
import imageio.v2 as imageio
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

bd = benchmark.get_benchmark_dict()
for suite_name, task_id, out in [("libero_10", 2, "kitchen_moka_task.png"),
                                 ("libero_90", 21, "kitchen_frying_pan_task.png")]:
    suite = bd[suite_name]()
    task = suite.get_task(task_id)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
    env.seed(0)
    env.reset()
    env.set_init_state(suite.get_task_init_states(task_id)[0])
    for _ in range(5):
        obs, _, _, _ = env.step([0.0] * 7)
    imageio.imwrite(out, obs["agentview_image"][::-1, ::-1])
    env.close()
    print(f"saved {out}  <-  {suite_name} task {task_id}: {task.language}")
