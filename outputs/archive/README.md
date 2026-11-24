# Historical experiments — not needed for the demo

This archive keeps the original experiments and evidence out of your demo workspace without deleting research records.

- `outputs/`: the complete earlier output tree, including all versions, raw comparisons, original initializations, selected models, datasets, job IDs, circuits, counts, usage ledgers, reports and ZIP bundles.
- `work/`: older build, analysis and verification scripts. The active Python environment was kept at the workspace's original `work/venv/` path.
- `relocation_index.json`: the old and new locations of moved top-level items.

The current demonstration is self-contained in `../demo/`. You do not need this folder during the presentation.

Relative relationships among historical output folders were retained. Some logs contain the absolute locations used when jobs were submitted; those remain unchanged as provenance. They are historical records, not instructions to recreate the old layout.

The IBM credential file was moved to the workspace's ignored `.secrets/` folder. It is not included in this archive or the demo. The historical IBM adapter can locate that private store if inspected later. The demo does not submit jobs or require the key.

No model training or IBM submission was performed during this cleanup.
