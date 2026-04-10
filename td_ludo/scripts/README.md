# td_ludo/scripts/

Canonical entry-point scripts for the td_ludo project.

This directory is the new home for invocation scripts. Each script here
is either a runpy wrapper around the legacy implementation at
`td_ludo/<name>.py`, or (eventually) a thin shim that imports `main()`
from the corresponding package module under `td_ludo.{eval,training,...}`.

During the refactor (Stage B8/B9 of REFACTOR_PLAN.md) the originals at
`td_ludo/<name>.py` keep working untouched, so production callers
(GCP sweep, training jobs) are not disrupted.

## Current contents

| Script | Status | Backing impl |
|---|---|---|
| init_v62_from_v61.py | runpy wrapper | td_ludo/init_v62_from_v61.py (legacy) |
