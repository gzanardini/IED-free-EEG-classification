# Single-Set Script Inconsistencies (Preserved in Refactor)

This refactor extracts common helper logic into `utils/single_set_common.py` but intentionally preserves behavior differences between scripts.

## Preserved differences

- Split strategy:
  - TUH single-set scripts use subject-level LOSO (`unique_subjects` folds).
  - EMC single-set scripts use sample-level LOOCV (one sample per fold).

- XGBoost defaults:
  - TUH single-set scripts keep `max_depth=7`, `subsample=0.8`, `learning_rate=0.05`.
  - EMC single-set scripts keep `max_depth=6`, `subsample=0.9`, `learning_rate=0.1`.

- Run count/config drift:
  - TUH single-set scripts keep `N_RUNS = 3`.
  - EMC single-set scripts keep `N_RUNS = 5`.

- Run naming / wandb init style:
  - TUH scripts keep `..._{combiner}_run_{run_n}` naming.
  - EMC IPS/background scripts keep `..._{combiner}run_{run_n}` naming.
  - Hypervent script keeps `RUN_NAME` construction and "Starting run" behavior.

- Dataset-specific guards:
  - TUH background variants keep the `gcc` invalid-combination skip.
  - No-IED TUH variants keep subject exclusion (`subject_to_skip`).

## Why these were not normalized

The requirement for this pass was behavior preservation. These differences affect training semantics, experiment bookkeeping, or historical output conventions. They are documented here for a future "normalization" pass if desired.
