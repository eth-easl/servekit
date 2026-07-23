# Quarantined: --export comma truncation

Jobs 75137 (`w_kv`) and 75138 (`all`) were submitted with
`--export=ALL,...,RELEASE_TAGS=weights,kv_cache,...`. SLURM splits `--export` on
commas, so `RELEASE_TAGS` truncated to `weights` and the two points silently ran as
duplicates of `w`. Their own logs prove it: `release_tags=weights`.

They are not wrong, just not what they claim: three weights-only runs, not a tag
sweep. Kept as extra samples of the `w` point, not as evidence about kv_cache.

Fixed by exporting into the environment and using a plain `--export=ALL`, plus a
guard in release_reload.sbatch that aborts if a `*_kv*`/`*all*` job receives a
single-tag RELEASE_TAGS.
