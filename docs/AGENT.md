# AGENT.md — How LLMs should work on SkyBrain
**Updated:** 2025-08-09

This file is the operating manual for *any* AI assistant (and humans) contributing to SkyBrain.

---

## 0) Mission
Make the original **Mavic Pro** feel new: safe, smooth, and smart. We **do not** replace DJI stabilization. We send **Virtual Stick** commands and add a policy layer with **Liquid Neural Networks (NavLNN + CineLNN)**.

---

## 1) Roles & responsibilities
- **Armand (owner):** approves scope, flies tests, sets safety limits.
- **Primary author LLM (ChatGPT):** writes code & docs per spec.
- **Review LLMs (Gemini, Claude):** review PRs (math/state estimation, Android lifecycle/concurrency, UI polish).

**Workflow:** single author → PR → reviews → merge. No parallel divergent codebases.

---

## 2) Repo map (what to edit and when)
- `app/` — Android/Kotlin app (SDK bridge, policy, UI).
- `training/` — Python (PyTorch) for LNN training + TFLite export.
- `docs/ARCHITECTURE-current.md` — **always** update when modules/interfaces change.
- `docs/ARCHITECTURE-planned.md` — roadmap and upcoming work.
- `docs/contracts/*` — **source of truth** for telemetry & model I/O. Update **in the same PR** when changing interfaces.
- `docs/model_cards/*` — NavLNN & CineLNN assumptions, inputs/outputs.
- `docs/SAFETY.md` — non‑negotiable rules; do **not** weaken.
- `docs/CHANGELOG.md` — every merged PR updates this.
- `docs/adrs/*` — one-page Architecture Decision Records for non-trivial choices.
- `.github/*` — PR template & issue templates.
- `tools/check.sh` — lint gate (extend with ktlint/detekt later).

---

## 3) Ground rules for LLM-generated code
1. **Read the context**: consume `ARCHITECTURE-current.md`, `contracts/*`, and relevant model cards before writing code.
2. **Never invent DJI APIs**. We are on **Mobile SDK v4.x** (Mavic Pro). No v5 symbols.
3. **No motor-level control**; commands are **body-frame velocities + yaw_rate** only.
4. **Safety-first**: KillSwitch, GeoFence, OA clamp, watchdogs must remain intact. Do not alter without explicit ADR + review.
5. **Do not bypass policy**: NavLNN outputs **caps** & **scan** decisions; CineLNN is advisory only. Neither overrides safety.
6. **Stick rate**: Virtual Stick **5–25 Hz**; default **25 Hz** sender. Don’t exceed or drop below spec.
7. **No blocking UI**: use coroutines/StateFlow; run control loops in a Foreground Service.
8. **Keep diffs small**: one feature per PR; update docs & tests in the same PR.

---

## 4) Prompt template for LLM contributors
> You are editing the SkyBrain repo for the Mavic Pro (DJI Mobile SDK v4.x). > First, read `docs/ARCHITECTURE-current.md`, `docs/contracts/*`, and `docs/SAFETY.md`. > Only propose changes that compile and respect the safety envelope. > When changing an interface, update `docs/contracts/*` and add an ADR in `docs/adrs/`. > Produce: (1) the diff (filenames + code), (2) any new tests, (3) doc updates, and (4) a short risk analysis. > Do not use MSDK v5 or non-existent APIs. Do not relax safety gates.

---

## 5) Common failure modes & how to avoid
- **Wrong App Key / package** → SDK registration fails. Ensure `com.armand.skybrain` matches the DJI dev console entry.
- **GO 4 grabbing USB** → close DJI GO 4; only one app can own the connection.
- **Side vs bottom USB on RC** → use **bottom** USB-A; unplug side cable.
- **Low command rate** → keep VS sender ≥ 20 Hz (we use 25 Hz); heartbeat watchdogs active.
- **Simulator confusion** → aircraft & RC powered, battery installed, props off; simulator doesn’t emulate obstacle cameras.
- **Minify/Proguard** → keep rules conservative for DJI & TFLite until we lock configs.

---

## 6) Change control
- **Safety, policy, or SDK bridge** changes require: ADR + 2 approvals (one LLM reviewer + Armand).
- **New model inputs/outputs** → update `contracts/policy_io.md` + model card; bump `schema_version` if added.
- **Breaking change**: increment app version; update CHANGELOG.

---

## 7) Testing ladder
1. **Unit**: policy math, clamps, rate limiters.
2. **Sim**: 2+ minutes with no watchdog trips; verify smooth hover/attitude changes.
3. **Field (props on, open area)**: ≤1.5 m/s, tilt ≤8°, verify KillSwitch & RTH.
4. **Night/sunset**: LightGovernor lowers speed; look‑then‑go policy observed.

---

## 8) Coding standards
- Kotlin: coroutines, StateFlow/SharedFlow; small, testable objects.
- Python: black + type hints; keep LTC cell small & exportable to TFLite.
- Logging: structured and rate-limited in control loops.

---

## 9) Feature requests: how to scope
- Provide: user story, acceptance tests, safety impact, updated contracts if needed.
- Large features (e.g., classifier, depth) land behind feature flags and off by default.

---

## 10) Absolutely do not
- Port to MSDK v5 for this aircraft.
- Send angles instead of body-frame velocities.
- Remove or weaken watchdogs, OA clamps, or geofences.
- Merge changes without updating docs/contracts.

---

## 11) Quick start for new contributors
1. Clone repo; open in Android Studio; sync Gradle.
2. Insert DJI App Key in Manifest.
3. Enable USB debugging; run on device.
4. Power RC+aircraft; start **DJISimulator**; verify VS @ 25 Hz.
5. Read this AGENT.md and `docs/ARCHITECTURE-current.md` before touching code.
