# snapper

A small CLI that drives the `docs/start_model.md` runbook: full deployment
lifecycle over `rcc`. Deployments are described by `deploy/*/snapper.toml`.

## Install

```bash
cd tools/snapper
uv pip install -e .        # or: pipx install -e .
```

## Commands

```bash
snapper list
snapper up   <dep> [--nodes N] [--variant V] [--no-probe] [--no-wait] [--set K=V]
snapper down <dep>
snapper status [<dep>]
snapper logs <dep> [--rank R] [-f]
snapper verify <dep>
```

`<dep>` is the `name` in a `deploy/<model>-<cluster>/snapper.toml`.

## Tests

```bash
cd tools/snapper
python3 -m pytest -v
```

The test suite does not need a cluster. It mocks the `rcc` boundary.

## End-to-end manual test

Per the repo rule to validate on compute nodes, confirm against a real
deployment once:

```bash
snapper up glm-47-flash-bristen --nodes 1
snapper status glm-47-flash-bristen
snapper verify glm-47-flash-bristen
snapper down  glm-47-flash-bristen
```

Expect the probe to pass, the serve job to reach ready, `verify` to return a
models list and a chat completion, and `down` to cancel the job.

## Snapshot seam

`--snapshot record|restore` is wired but inert while every manifest has
`[snapshot].enabled = false`. It will activate once the `snapshot/` subsystem
lands a real record/restore serve path.
