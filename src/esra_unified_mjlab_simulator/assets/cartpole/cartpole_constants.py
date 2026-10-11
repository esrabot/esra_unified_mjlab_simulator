from pathlib import Path
import mujoco

from mjlab.entity import Entity, EntityCfg, EntityArticulationInfoCfg
from mjlab.actuator import XmlActuatorCfg


CARTPOLE_XML: Path = Path(__file__).parent / "xmls" / "cartpole.xml"
assert CARTPOLE_XML.exists(), f"XML not found: {CARTPOLE_XML}"


def get_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(CARTPOLE_XML))

def get_cartpole_robot_cfg() -> EntityCfg:
    """Get a fresh CartPole robot configuration instance."""
    actuators = (
        XmlActuatorCfg(
        target_names_expr=("slide",),
        ),
    )
    articulation = EntityArticulationInfoCfg(actuators=actuators)
    return EntityCfg(
        spec_fn=get_spec,
        articulation=articulation
    )
