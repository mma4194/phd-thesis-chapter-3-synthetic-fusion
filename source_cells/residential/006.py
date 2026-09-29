# %% CELL 3.PERSIST — persist chronological splits for EXT audits (run once)
for name, d in [("REAL_TRAIN", df_tr), ("REAL_VAL", df_va), ("REAL_TEST", df_te)]:
    p = os.path.join(OUT_SYN, f"{name}_SPLIT.parquet")
    d.to_parquet(p, index=False)
    print(name, d.shape, "->", p)