.PHONY: format
format:
	uv run ruff format
	uv run ruff check --fix

.PHONY: type
type:
	uv run ty check
	uv run pyright

.PHONY: stubs
stubs:
	bash typings/generate_mujoco_stubs.sh

.PHONY: check
check: format type
