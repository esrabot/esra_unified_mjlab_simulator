"""CCBR Leo robot velocity environment configurations."""


from esra_unified_mjlab_simulator.assets.leo_quadruped.leo_constants import (
    LEO_ACTION_SCALE,
    get_leo_robot_cfg,
    get_leo_robot_cfg_learned,
)

from esra_unified_mjlab_simulator.tasks.velocity.env.velocity_env import make_velocity_env_cfg

from mjlab.envs import ManagerBasedRlEnvCfg
import mjlab.envs.mdp as mj_mdp
from esra_unified_mjlab_simulator.tasks.velocity.env import mdp

from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers import TerminationTermCfg

from mjlab.sensor import (
    ContactMatch,
    ContactSensorCfg,
    GridPatternCfg,
    ObjRef,
    RayCastSensorCfg,
    RingPatternCfg,
    TerrainHeightSensorCfg,
)


def make_ccbr_leo_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create CCBR Leo robot rough terrain velocity configuration."""
    cfg = make_velocity_env_cfg()

    ##
    # Simulation
    ##
    cfg.sim.mujoco.ccd_iterations = 500
    cfg.sim.mujoco.impratio = 10
    cfg.sim.mujoco.cone = "elliptic"
    cfg.sim.contact_sensor_maxmatch = 500

    ##
    # Entities
    ##
    cfg.scene.entities = {"robot": get_leo_robot_cfg()}

    ##
    # Sensors
    ##

    # Leo robot uses Ball_End_Effector collision geoms as foot contacts
    # Need to be in order of definition in XML.
    geom_names = (
        "Ball_End_Effector_collision",  # back_left
        "Ball_End_Effector_2_collision",  # back_right
        "Ball_End_Effector_4_collision",  # front_right
        "Ball_End_Effector_3_collision",  # front_left
    )
    site_names = ("BL", "BR", "FR", "FL")

    # Set raycast sensor frame to Leo base.
    for sensor in cfg.scene.sensors or ():
        if sensor.name == "terrain_scan":
            assert isinstance(sensor, RayCastSensorCfg)
            assert isinstance(sensor.frame, ObjRef)
            sensor.frame.name = "base"

    # Wire foot height scan to per-foot sites.
    for sensor in cfg.scene.sensors or ():
        if sensor.name == "foot_height_scan":
            assert isinstance(sensor, TerrainHeightSensorCfg)
            sensor.frame = tuple(
                ObjRef(type="site", name=s, entity="robot") for s in site_names
            )
            sensor.pattern = RingPatternCfg.single_ring(
                radius=0.04, num_samples=4)

    feet_ground_contact = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(mode="geom", pattern=geom_names, entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )

    nonfoot_ground_cfg = ContactSensorCfg(
        name="nonfoot_ground_touch",
        primary=ContactMatch(mode="geom", entity="robot",
                            pattern=r".*_collision$", exclude=tuple(geom_names)),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found",), reduce="none", num_slots=1,
    )

    cfg.scene.sensors = (cfg.scene.sensors or ()) + (
        feet_ground_contact,
        nonfoot_ground_cfg,
    )

    ##
    # Terrains
    ##
    if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = True

    ##
    # Actions
    ##
    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, mj_mdp.JointPositionActionCfg)
    joint_pos_action.scale = LEO_ACTION_SCALE

    ##
    # Viewer
    ##
    cfg.viewer.body_name = "base"
    cfg.viewer.distance = 1.5
    cfg.viewer.elevation = -10.0

    ##
    # Observations
    ##

    ##
    # Rewards
    ##
    # Update joint name patterns for leo robot naming convention
    cfg.rewards["pose"].params["std_standing"] = {
        r"^(back|front)_(left|right)_hip_(roll|pitch)$": 0.05,
        r"^(back|front)_(left|right)_knee_pitch$": 0.1,
    }
    cfg.rewards["pose"].params["std_walking"] = {
        r"^(back|front)_(left|right)_hip_(roll|pitch)$": 0.3,
        r"^(back|front)_(left|right)_knee_pitch$": 0.6,
    }
    cfg.rewards["pose"].params["std_running"] = {
        r"^(back|front)_(left|right)_hip_(roll|pitch)$": 0.3,
        r"^(back|front)_(left|right)_knee_pitch$": 0.6,
    }
    cfg.rewards["pose"].weight = 0.25

    cfg.rewards["upright"].params["asset_cfg"].body_names = ("base",)
    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("base",)

    # Configure foot site-based rewards with foot sites
    for reward_name in ["foot_clearance", "foot_slip"]:
        if reward_name in cfg.rewards:
            cfg.rewards[reward_name].params["asset_cfg"].site_names = site_names

    cfg.rewards["body_ang_vel"].weight = 0.0
    cfg.rewards["angular_momentum"].weight = 0.0
    cfg.rewards["air_time"].weight = 0.0

    ##
    # Events
    ##
    cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names

    ##
    # Terminations
    ##
    cfg.terminations["illegal_contact"] = TerminationTermCfg(
        func=mdp.illegal_contact,
        params={"sensor_name": nonfoot_ground_cfg.name},
    )

    ##
    # Apply play mode overrides
    ##
    if play:
        # Effectively infinite episode length.
        cfg.episode_length_s = int(1e9)

        cfg.observations["actor"].enable_corruption = False

        assert cfg.commands is not None
        assert type(cfg.commands["twist"]) is mdp.UniformVelocityCommandCfg
        cfg.commands["twist"].ranges.lin_vel_x = (-0.0, 0.0)
        cfg.commands["twist"].ranges.lin_vel_y = (-0.0, 0.0)
        cfg.commands["twist"].resampling_time_range = (1.0, 3.0)

        cfg.terminations.pop("out_of_terrain_bounds", None)

        cfg.curriculum = {}

        cfg.events.pop("push_robot", None)
        cfg.events["randomize_terrain"] = EventTermCfg(
            func=mj_mdp.randomize_terrain,
            mode="reset",
            params={},
        )

        if cfg.scene.terrain is not None:
            if cfg.scene.terrain.terrain_generator is not None:
                cfg.scene.terrain.terrain_generator.curriculum = False
                cfg.scene.terrain.terrain_generator.num_cols = 5
                cfg.scene.terrain.terrain_generator.num_rows = 5
                cfg.scene.terrain.terrain_generator.border_width = 10.0

    return cfg
