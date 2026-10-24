# README

## Environment Setup

Install uv (a package manager). Using a package manager such as Homebrew on macOS makes installation easier.

Then run `uv sync` to set up the environment based on `pyproject.toml` and `uv.lock`.

One advantage of uv is that you can update dependencies through commands without manually editing `pyproject.toml` or `uv.lock`.

## Entry Points

1. View the environment:

   ```bash
   uv run mjpython src/esra_unified_mjlab_simulator/scripts/view_env.py esra_unified_mjlab_simulator.tasks.velocity.config.leo_quadruped.env_cfg:make_ccbr_leo_rough_env_cfg
   ```

   Then open [http://localhost:8080](http://localhost:8080) in your browser.

2. Cartpole (a toy task with a pole mounted on a cart; the cart is trained to move while keeping the pole upright):

   ```bash
   uv run mjpython src/esra_unified_mjlab_simulator/scripts/view_env.py esra_unified_mjlab_simulator.tasks.cartpole.env_cfg:cartpole_env_cfg
   ```

   Then open [http://localhost:8080](http://localhost:8080) in your browser.

3. `bash typings/generate_mujoco_stubs.sh` (for editor autocompletion).

There are currently no scripts for the other parts.

## Directory Responsibilities

`src/esra_unified_mjlab_simulator/assets` contains the Leo and Cartpole model files.

The following are quick-start notes I had AI help organize. I am not sure whether they are correct.

### src/esra_unified_mjlab_simulator/tasks/cartpole/

**Concepts used in env_cfg.py:**

- `SceneCfg`: What exists in the world. For example, the world contains an entity named `robot` using the Cartpole model, with 64 copies of the environment running simultaneously.
- `SimulationCfg`: How numerical simulation is performed.
- `JointVelocityActionCfg`: How PPO output values become cart control commands. This includes which entity and actuators (joints, motors) to control, and the scale (for example, multiplying PPO output by 20 to obtain a target velocity in m/s).
- `ObservationTermCfg`: What the model receives as input, such as reading part of the `env.sim.data.qpos` tensor.
- `RewardTermCfg`: Defines rewards.
- `TerminationTermCfg`: Defines when an episode ends.
- `EventTermCfg`: Defines random events at reset or during an episode.
- `ManagerBasedRlEnvCfg`: Combines the above.
- `ViewerConfig`: Configures the camera.

**rl_cfg.py**

- `RslRlModelCfg`: Model architecture.
- `RslRlPpoAlgorithmCfg`: PPO hyperparameters.
- `RslRlOnPolicyRunnerCfg`: Combines the above and specifies other hyperparameters.

### src/esra_unified_mjlab_simulator/tasks/velocity/

#### src/esra_unified_mjlab_simulator/tasks/velocity/config/leo_quadruped/algo_cfg.py

Similar to `src/esra_unified_mjlab_simulator/tasks/cartpole/rl_cfg.py`.

#### env/velocity_env.py

Defines a single large function, `make_velocity_env_cfg()`. It imports `src/esra_unified_mjlab_simulator/tasks/velocity/env/mdp` to use various custom functions.

**Custom Sensors**

- `terrain_scan`: Creates a `RayCastSensorCfg` with this name—a distance sensor that measures ground elevations on a grid around the robot. `ObjRef` specifies the robot body as the reference.
- `foot_height_scan`: Creates a `TerrainHeightSensorCfg` with this name to determine how high the feet are above the ground.

They are included through `sensors=(terrain_scan, foot_height_scan)` in `SceneCfg`, then accessed using parameters such as `params={"sensor_name": "terrain_scan"}`. Markov Decision Process.

**Observations**

`actor_terms` / `critic_terms`: What the actor and critic can observe. Noise is added to the actor’s observations to account for inaccuracies in real sensors.

**Actions:** Convert policy output values into target positions for the robot’s joints.

**Commands: Set locomotion targets for the robot.**

The environment randomly generates commands specifying forward/backward velocity, lateral velocity, and turning velocity. For example, forward/backward velocity is sampled from −1 to 1 m/s. Special requirements are then applied with certain probabilities—for example, a 10% chance of requesting the robot to stand still, setting all target velocities to zero. Each command lasts a random 3–8 simulation seconds before being regenerated. This command is named `"twist"` and is included in the actor’s input so the policy knows how it is currently expected to move.

**Curriculum: Gradually increase training difficulty.**

- `"terrain_levels"`: If `"twist"` requests forward movement and the robot travels far enough, it moves to harder terrain; otherwise, it moves to easier terrain.
- `"command_vel"`: Initially, `"twist"` can request forward speeds up to 1 m/s. At 120,000 steps, this increases to 2 m/s. At 240,000 steps, it increases to 3 m/s.

**Rewards:** PPO rewards. `params` contains parameters we define. `command_name` identifies the reference target, and the function returns `torch.exp(-lin_vel_error / std**2)`. Example: `{"command_name": "twist", "std": math.sqrt(0.25)}`.

**Events:** Random events affecting the model.

**Terminations:** When an episode ends.

**Metrics:** Other statistics that do not contribute to the loss.

Finally, everything is assembled using `ManagerBasedRlEnvCfg`.

#### src/esra_unified_mjlab_simulator/tasks/velocity/runner/velocity_runner.py

Extends the original `MjlabOnPolicyRunner` (which organizes training iterations) to export ONNX files when saving. ONNX is similar to `.pt` but is convenient for cross-platform use. If W&B is being used and `upload_model` is enabled, the file is uploaded to W&B, a website for storing and viewing model training records.

#### src/esra_unified_mjlab_simulator/tasks/velocity/config/leo_quadruped/env_cfg.py

Uses `src/esra_unified_mjlab_simulator/tasks/velocity/env/velocity_env.py` and adds Leo-specific parameters.

- `terrain_scan`: Sets `frame=ObjRef(type="body", name="base", entity="robot")`.
- `foot_height_scan`: Uses `ObjRef(type="site", name=s, entity="robot") for s in site_names` to specify Leo’s feet.

Custom sensors: `feet_ground_contact` / `nonfoot_ground_cfg`, configured using `ContactSensorCfg`. `primary` and `secondary` specify the two sides of the contact.

```xml
<body name="foot">
    <geom name="foot_ball" type="sphere" size="0.03"/>
</body>
```

- The body name is `"foot"`: the foot component.
- The geom name is `"foot_ball"`: the spherical collision shape on the foot.

`mode=geom or body`

`fields=("found", "force")`: Records whether contact is detected and the contact force. With `reduce="netforce"`, a foot may have multiple contact points, and `"netforce"` combines their forces into one resultant force.