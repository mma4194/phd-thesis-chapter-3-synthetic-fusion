# %% CELL 14.6.PRE — Protocol base path compatibility for report-only A0 Q4 materialization
# Purpose:
#   Cell 14.6 may use an in-memory protocol base and later export
#   CELL14_6_PRE_Q4_PROTOCOL_BASE_PATH. If the base was in memory, that variable
#   may not exist. This cell defines a safe placeholder.
#
# Safety:
#   - Does not mutate data.
#   - Does not promote Q4.
#   - Does not use TEST for decisions.

from pathlib import Path

def _cell146pre_log(msg):
    print(f"[Cell14.6.PRE] {msg}")

if "CELL14_6_PRE_Q4_PROTOCOL_BASE_PATH" not in globals():
    candidate_paths = [
        globals().get("PROTOCOL_SYN_TEST_130_PATH"),
        globals().get("PROTOCOL_SYN_TEST_PATH"),
        globals().get("A2_PROTOCOL_FINAL_PATH"),
        globals().get("A2_PROTOCOL_TEST_PATH"),
    ]

    resolved = None

    for p in candidate_paths:
        if p is None:
            continue

        pp = Path(str(p)).expanduser()

        if pp.exists():
            resolved = str(pp.resolve())
            break

    if resolved is None:
        resolved = "<in_memory>"

    CELL14_6_PRE_Q4_PROTOCOL_BASE_PATH = resolved

globals()["CELL14_6_PRE_Q4_PROTOCOL_BASE_PATH"] = CELL14_6_PRE_Q4_PROTOCOL_BASE_PATH

_cell146pre_log(f"CELL14_6_PRE_Q4_PROTOCOL_BASE_PATH = {CELL14_6_PRE_Q4_PROTOCOL_BASE_PATH}")
_cell146pre_log("PASS: Cell 14.6 protocol base path compatibility variable is defined.")