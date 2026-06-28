# snapper — model lifecycle CLI

**Status:** Design approved 2026-06-27
**Author:** Xiaozhe Yao (with Claude)
**Related:** `docs/start_model.md` (the human runbook this tool automates),
`.agents/skills/deploy/SKILL.md` (the dir generator), `.rcc/config.toml` (rcc profiles)

## 1. Purpose

`snapper` is a small Python CLI that manages the **full lifecycle** of an LLM
deployment on the CSCS clusters (bristen / beverin / clariden), turning the
manual `docs/start_model.md` runbook into commands.

Today, "spinning up a model" is a manual chain:

```
rcc push → rcc run "sbatch probe…" → read log → rcc run "sbatch serve…"
         → wait for ready markers → grep last_service.env → curl
```

`snapper` collapses that into `snapper up <deployment>` and adds
`down / status / logs / verify` for the rest of the lifecycle. It also reserves
a first-class **snapshot seam** (opt-in fast cold start via this repo's
snapshot/restore subsystem), wired end-to-end but inert until the snapshot
restore path — "N5b" — is proven.

### Goals

- One command to bring a model up end-to-end, mirroring the runbook exactly.
- Manage a running deployment: `down`, `status`, `logs`, `verify`.
- Discover deployments automatically from the repo.
- Keep the snapshot integration as a wired-but-inert seam, so enabling it later
  is config + a serve script, not a snapper rewrite.

### Non-goals (v1)

- **Generating** new `deploy/<model>-<cluster>/` dirs. That stays with the
  `deploy` agent skill. snapper *drives* existing deployments.
- Re-implementing SSH/rsync. `rcc` owns transport; snapper shells out to it.
- Implementing snapshot record/restore logic. That lives in the `snapshot/`
  subsystem (LD_PRELOAD interposers + serve script). snapper only injects env /
  selects a script.
- A local state database. Runtime state is derived from the cluster.

## 2. Foundational decisions

These were settled during brainstorming and frame the whole design:

| Decision | Choice |
|----------|--------|
| Scope | Full lifecycle manager (`up`/`down`/`status`/`logs`/`verify`) + snapshot fast-launch seam |
| Snapshot integration | Design the seam now; vanilla path works first; snapshot opt-in and inert in v1 |
| Stack | Python package that shells out to `rcc` for transport |
| Deployment metadata | Per-deploy `snapper.toml` inside each `deploy/<model>-<cluster>/`; discovered by scanning |
| Runtime state | Derived from the cluster (`squeue` + `last_service.env`); no local state file |
| `up` blocking model | Synchronous (wait until ready) by default; `--no-wait` to return after submit |

## 3. Architecture

`snapper`'s **only** I/O boundary to the cluster is `rcc`. Everything else is
pure, unit-testable logic. This single boundary is what makes the tool testable
without a cluster: mock `rcc.py`, feed fixtures to the parsers.

```
tools/snapper/
  pyproject.toml          # console_scripts: snapper = snapper.cli:main
  snapper/
    __init__.py
    cli.py                # Typer app + subcommands (thin; arg parsing only)
    config.py             # discover & parse snapper.toml → Deployment dataclass
    rcc.py                # the ONLY shell-out: push(), run(), run_capture()
    slurm.py              # submit→parse jobid, poll squeue, wait_for_ready (grep log)
    service.py            # read last_service.env, verify (curl)
    snapshot.py           # the seam: build env/variant for record|restore (inert in v1)
  tests/                  # mock rcc.py; fixture-driven parsers
  README.md               # install + the documented end-to-end manual test
```

Location: `tools/snapper/` (not a top-level `snapper/`) to avoid visual
collision with the existing `snapshot/` tree.

### Module responsibilities

- **`cli.py`** — Typer app. One function per subcommand. Resolves the deployment
  via `config`, calls into `slurm`/`service`/`snapshot`, formats output. No
  business logic beyond orchestration.
- **`config.py`** — `discover()` scans `deploy/*/snapper.toml`; `load(name)`
  returns a `Deployment` dataclass (fields below). Validates required keys, gives
  a clear error listing known deployments when a name is unknown.
- **`rcc.py`** — the single shell-out boundary. `push(profile)`,
  `run(profile, cmd) -> int`, `run_capture(profile, cmd) -> (rc, stdout, stderr)`.
  Builds `rcc --profile <p> run bash -lc "<cmd>"`. Every other module talks to
  the cluster *through here* so tests mock exactly one thing.
- **`slurm.py`** — pure-ish helpers: `parse_jobid(sbatch_stdout)`,
  `squeue_state(profile, jobid)`, `submit(deployment, nodes, env, script)`,
  `wait_for_ready(deployment, jobid, nodes, timeout)`. The parsers are pure
  functions over text; only `submit`/`wait` touch `rcc`.
- **`service.py`** — `read_last_service_env(profile, remote_dir)`,
  `verify(deployment, endpoint)` (curl `/v1/models` + one chat/generate request).
- **`snapshot.py`** — `plan(deployment, mode) -> (extra_env, serve_script)`.
  In v1, returns no-op when `[snapshot].enabled` is false.

## 4. Deployment manifest — `snapper.toml`

One per `deploy/<model>-<cluster>/`. snapper discovers deployments by scanning
`deploy/*/snapper.toml`. The static metadata snapper needs (and `rcc` does not
store) lives here.

```toml
name    = "glm-47-flash-bristen"        # snapper handle (default: dir basename)
profile = "glm-47-flash-bristen"        # rcc profile → host + remote_dir
engine  = "sglang"                      # sglang | vllm
served_model_name = "zai-org/GLM-4.7-Flash"
port = 8080
default_nodes = 5

[scripts]
probe = "probe_sglang.sbatch"
serve = "serve_glm_47_flash_sglang.sbatch"

[ready]                                  # grepped per-rank to declare a rank up
markers = ["fired up and ready to roll", "vmagent_started"]
log_pattern = "logs/opentela-{job}-{rank}.log"

[verify]
models_path = "/v1/models"
chat_path   = "/v1/chat/completions"     # vLLM/SGLang OpenAI-compatible

[variants.router]                        # optional: alternate serve scripts
serve = "serve_glm_47_flash_sglang_router.sbatch"
[variants.dp2]
serve = "serve_glm_47_flash_sglang_router_dp2.sbatch"

[snapshot]                               # the seam — inert in v1
enabled = false
# record_env  = { SNAPSHOT_MODE = "record" }
# restore_env = { SNAPSHOT_MODE = "restore" }
# serve       = "serve_..._snapshot.sbatch"   # optional snapshot-specific script
```

### `Deployment` dataclass (parsed form)

| Field | Source | Notes |
|-------|--------|-------|
| `name` | `name` or dir basename | snapper handle, unique |
| `profile` | `profile` | resolves host + remote_dir via rcc |
| `engine` | `engine` | `sglang` \| `vllm` |
| `served_model_name` | `served_model_name` | for `verify` |
| `port` | `port` | engine/endpoint port |
| `default_nodes` | `default_nodes` | overridable with `--nodes` |
| `probe_script` | `[scripts].probe` | optional; absent → `--no-probe` implied |
| `serve_script` | `[scripts].serve` | default serve script |
| `variants` | `[variants.*]` | name → serve script override |
| `ready_markers` | `[ready].markers` | all must appear per rank |
| `log_pattern` | `[ready].log_pattern` | `{job}`,`{rank}` substituted |
| `verify_paths` | `[verify]` | models/chat paths |
| `snapshot` | `[snapshot]` | enabled flag + record/restore env + optional script |

The **router / dp2** cases need no special handling: `glm-47-flash-bristen-xzyao`
is already its own dir + profile (its own deployment), and its in-dir alternate
serve scripts become `[variants.*]`, selected with `snapper up <dep> --variant dp2`.

## 5. Command surface

```
snapper list                       # discovered deployments + cheap running/idle state
snapper up   <dep> [--nodes N] [--variant V] [--no-probe] [--no-wait]
                   [--no-push] [--snapshot record|restore] [--set KEY=VAL ...]
snapper down <dep>                 # scancel the running job
snapper status [<dep>]             # squeue + last_service.env; all deployments if omitted
snapper logs <dep> [--rank R] [--follow]
snapper verify <dep>               # curl /v1/models + one chat/generate smoke request
```

- `<dep>` is the `snapper.toml` `name` (default: deploy-dir basename).
- `--set KEY=VAL` injects env before `sbatch` (e.g. `--set TP_SIZE=2`), matching
  the runbook's `TP_SIZE=2 sbatch …` override pattern.

## 6. `up` orchestration

Mirrors the runbook: **push → probe(+wait+check) → serve → wait-for-ready →
print endpoint.**

1. Resolve deployment (`config.load`) + rcc profile.
2. Unless `--no-push`: `rcc --profile <p> push`.
3. Unless `--no-probe` (or no probe script): submit probe, poll `squeue` until
   the job leaves the queue, scan the probe log for failure markers. **On probe
   failure, abort before serve** and print the probe log tail + a pointer to
   `.agents/skills/deploy/references/gotchas.md` (honors "always probe before
   serve").
4. Resolve serve script: `--variant V` → `[variants.V].serve`; else
   `[snapshot].serve` if `--snapshot` set and defined; else `[scripts].serve`.
   Compute env = `--set` pairs + `snapshot.plan(...)` env.
5. Submit serve with `--nodes=N` **on the CLI** (Slurm honors the directive
   override, not an env var — per the runbook warning). Parse the jobid.
6. Unless `--no-wait`: poll until **all N ranks** show every `[ready].marker` in
   their per-rank log (path from `log_pattern`), with a generous timeout (cold
   start is 10–20 min; default ~30 min, overridable). Emit live progress
   (`rank 2/5 ready …`). On timeout: print log tails, **leave the job running**,
   and instruct the user to use `snapper logs` / `snapper down`.
7. Read `last_service.env`, print endpoints + a ready-to-paste `verify` curl.

**Blocking model:** `up` is synchronous (waits until ready) by default, because
"spin up a model" implies you want it serving. `--no-wait` returns immediately
after submit with the jobid.

## 7. Runtime state — no local state file

`status` / `down` / `logs` derive everything from the cluster, so nothing can go
stale:

- **jobid** comes from `last_service.env` (see the additive change below);
  fallback is `squeue -u $USER` matched by the serve script's `#SBATCH --job-name`.
- **`status`** = `cat last_service.env` + a filtered `squeue` (the runbook's
  status-helper formula), per deployment or across all discovered ones.
- **`logs`** = `rcc run "tail [-f] -n N <log_pattern>"` for the requested rank
  (default rank 0); `--follow` streams via `rcc run`.
- **`down`** = `scancel <jobid>` (no-op with a friendly message if nothing runs).

### Required additive change to serve scripts

Have each serve script also write `JOBID=$SLURM_JOB_ID` into `last_service.env`
(it already writes `ENDPOINT_*`). This gives `down`/`logs` a direct target
instead of guessing. It is the **only** edit snapper needs to existing deploy
artifacts; the squeue-by-job-name fallback covers scripts not yet updated.

## 8. Snapshot seam (inert in v1)

`snapshot.py::plan(deployment, mode)` turns `--snapshot record|restore` into:

- **extra env** (e.g. `SNAPSHOT_MODE=record`) injected before `sbatch`, and/or
- a **snapshot-specific serve script** from `[snapshot].serve`.

In v1, every deployment has `[snapshot].enabled = false`, so `plan` is a no-op
and `--snapshot` is wired end-to-end but does nothing — until the `snapshot/`
subsystem lands a real record/restore serve path (N5b). snapper stays agnostic
to snapshot internals: it injects env / picks a script and nothing more.

## 9. Error handling

| Condition | Behavior |
|-----------|----------|
| Unknown deployment name | Error listing all discovered deployments |
| Missing/invalid `snapper.toml` keys | Validation error naming the field |
| `rcc` non-zero exit | Surface stderr, abort the command |
| Probe failure | Abort serve, print probe log tail + gotchas.md pointer |
| Ready timeout | Print log tails, **leave job running**, suggest `logs`/`down` |
| `down` with nothing running | Friendly no-op message |
| `verify` connection refused | Report not-ready; suggest `status`/`logs` |

## 10. Testing

- **Unit (no cluster):** `rcc.py` is the single mock point. Pure parsers —
  sbatch jobid, `squeue` state, `last_service.env`, ready-marker scan,
  `snapper.toml` load — are tested with fixtures. This is the bulk of coverage
  and runs in CI.
- **Integration (mocked rcc):** drive `up`/`down`/`status` with a fake `rcc`
  that returns canned cluster responses; assert the command sequence and
  control flow (probe-fail aborts serve, timeout leaves job running, etc.).
- **End-to-end (manual, documented):** a `tools/snapper/README.md` runbook for a
  real `snapper up glm-47-flash-bristen` on the cluster, per the repo rule to
  validate on compute nodes — not asserted in CI.

Dependencies stay minimal: `typer` + stdlib `tomllib` (3.11+); `rich` optional
for progress output.

## 11. Rollout

1. Build the package + `up`/`down`/`status`/`logs`/`verify` over the vanilla
   path, with the snapshot seam inert.
2. Add a `snapper.toml` to each existing `deploy/<model>-<cluster>/` dir and the
   `JOBID=` line to its serve script.
3. Document install + the manual e2e test in `tools/snapper/README.md`; add a
   short pointer from `docs/start_model.md`.
4. (Later, out of scope) flip `[snapshot].enabled` once N5b proves restore.
