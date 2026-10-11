"""Custom mjlab tasks. Imported by mjlab via the ``mjlab.tasks`` entry point."""

from pathlib import Path

# Must be defined before the task imports: asset modules import it from here.
ESRA_MJLAB_SRC_PATH: Path = Path(__file__).parent

from esra_unified_mjlab_simulator.tasks import cartpole as cartpole  # noqa: E402
from esra_unified_mjlab_simulator.tasks import velocity as velocity  # noqa: E402
