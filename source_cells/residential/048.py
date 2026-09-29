# ==========================================================
# CLEANUP AFTER CELL 12.b v27.1.1
# Remove stale DDPM globals from earlier notebook runs
# ==========================================================

import gc

_STALE_12B_DDPM_GLOBALS = [
    "syn_ddpm_df",
    "syn_ddpm_val_df",
    "ddpm_model",
    "ddpm",
    "scaler",
    "ddpm_train_df",
    "ddpm_val_df",
    "ddpm_test_real_df",
    "M_tr",
    "M_val",
    "M_te",
    "train_medians",
    "ddpm_train_imp",
    "ddpm_val_imp",
    "Xdd_tr",
    "Xdd_val",
    "Xdd_te_seed",
    "HIST_TR",
    "HIST_VAL",
    "DRV_TR_ARR",
    "DRV_VAL_ARR",
    "DRV_TE_ARR",
    "BIN_TR_ARR",
    "BIN_VAL_ARR",
    "BIN_TE_ARR",
    "REGIME_TR_ARR",
    "REGIME_VAL_ARR",
    "REGIME_TE_ARR",
    "SEG_LEN",
    "STRIDE",
    "TR_SEG_STARTS",
    "train_ds",
    "train_loader",
    "opt",
    "DDPM_COLUMN_CALIBRATION",
    "DDPM_COLUMN_CALIBRATION_DF",
    "DDPM_COLUMN_CALIBRATION_EXPORT",
]

_removed = []
for _name in _STALE_12B_DDPM_GLOBALS:
    if _name in globals():
        del globals()[_name]
        _removed.append(_name)

gc.collect()

if "torch" in globals() and torch.cuda.is_available():
    torch.cuda.empty_cache()

_survivors = [_name for _name in _STALE_12B_DDPM_GLOBALS if _name in globals()]
if _survivors:
    raise RuntimeError(
        "[POST-12b-CLEANUP] Stale DDPM globals survived cleanup: "
        f"{_survivors}"
    )

log(f"[POST-12b-CLEANUP] Removed stale DDPM globals: {_removed}")
log("[POST-12b-CLEANUP] Verified no stale DDPM globals remain.")
log("[POST-12b-CLEANUP] Safe to proceed to modular Cell 12.c.0.")