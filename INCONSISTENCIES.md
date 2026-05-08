# Training Script Inconsistencies

This note reflects the current state of the codebase after the move to
`utils/model_training.py`. Several differences that existed earlier have now
been normalized, while a smaller set of wrapper-level inconsistencies still
remains.

## Resolved

- Split strategy:
  - Single-set training now uses subject-level LOSO for both TUH and EMC.
  - The earlier EMC sample-level LOOCV behavior is no longer present in the
    active training path.

- Run naming / wandb init style:
  - The active training wrappers now use the shared naming flow from
    `utils/model_training.py`.
  - The earlier TUH vs EMC naming mismatch and the hypervent-specific
    single-set run-name handling are no longer part of the active code path.

## Still Present

- XGBoost defaults:
  - TUH single-set wrappers still use `max_depth=7`, `subsample=0.8`,
    `learning_rate=0.05`.
  - EMC single-set wrappers still use `max_depth=6`, `subsample=0.9`,
    `learning_rate=0.1`.

- Run count/config drift:
  - EMC single-set wrappers still use `N_RUNS = 5`.
  - TUH single-set wrappers are not fully standardized: some use
    `N_RUNS = 3`, while others use `N_RUNS = 5`.

- Dataset-specific guards:
  - TUH background variants still keep the `gcc` invalid-combination skip.
  - TUH no-IED variants still keep subject exclusion through
    `subjects_to_skip`.

## Interpretation

The major behavioral inconsistency has been removed: TUH and EMC single-set
training now follow the same LOSO split logic through the shared module.

What remains is mostly wrapper-level configuration drift rather than core
pipeline divergence. If we want a final normalization pass, the next targets
would be:

- standardizing `N_RUNS` across wrappers
- deciding whether TUH and EMC should keep different XGBoost defaults
- reviewing whether dataset-specific guards should stay wrapper-specific or be
  encoded more declaratively in shared config
