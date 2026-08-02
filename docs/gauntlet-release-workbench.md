# V1.0 Gauntlet Release Workbench

This file is the durable state for the final builder/critic release loop. It
records the original objective, the evidence required to pass, and the
boundaries that prevent a visual-polish run from silently changing the driver.

## Objective

Ship Self-Driving-RL v1.0 as a calm, professional, visually coherent learning
project. Preserve the useful driving telemetry, make the native Pygame viewer
genuinely pleasant to watch, provide a browser-accessible demonstration of an
actual frozen-policy trajectory, and package the code and models for GitHub and
Hugging Face.

## Frozen behavior boundary

- Checkpoint before visual work: `10b9c67`.
- Do not retrain, replace, or retune the recommended driver during this loop.
- Do not weaken collision geometry or the longitudinal/lane safety shields.
- Preserve the final hard/dynamic holdout: 100/100 completions, zero crashes,
  seed 690000, recorded in
  `runs/evaluation/v8-braking-final-unseen-100.json`.
- A browser replay must be labelled honestly as a deterministic trajectory
  exported from the real policy; it must not imply that Torch inference runs in
  the visitor's browser unless that is actually implemented.

## Release gates

### Native presentation

- The running 1440x810 viewer has a calm, coherent visual hierarchy.
- Driving remains the focal point; telemetry supports rather than surrounds it.
- The stats currently used to understand the agent remain available.
- No clipped, overlapping, unreadable, or low-contrast text.
- A fresh visual critic scores the integrated result at least 8/10 with no
  critical usability defect.

### Browser demonstration

- A static build opens without a Python server and visibly plays an exported
  trained-agent route.
- Desktop and mobile layouts are usable.
- Pause/resume, speed, route restart, and explanatory telemetry work.
- The page clearly distinguishes model facts, measured results, and limitations.
- The exported trajectory has a documented regeneration command and provenance.

### Engineering and ML reproducibility

- Python tests and Ruff pass.
- Browser lint/type/build checks pass, plus an automated browser smoke test.
- A clean-clone install/run path is documented and tested as far as the local
  environment permits.
- Recommended model, policy layers, observation/action shapes, training method,
  evaluation seeds, results, limitations, and artifact hashes are documented.
- Repository and release artifacts contain no credentials or personal paths.

### Publication

- GitHub README, license, browser demo, release notes, and media are ready.
- A Hugging Face model card and an explicit artifact manifest are ready.
- No public upload, deployment, release creation, or Reddit submission happens
  without the user's approval of the final inspected artifacts.

## Stop conditions

Stop when every gate passes, when two consecutive critic rounds identify no new
high-impact fix, or when a blocker needs credentials, external configuration, or
human aesthetic judgment. Preserve a rollback commit before each major wave.

## Round log

### Baseline

- Native screenshot: `runs/gauntlet/baseline-pygame.png`.
- Full Python suite: passed (97 tests).
- Ruff: passed.
- Independent visual score: 3.5/10. The useful simulator was visually dense,
  neon-heavy, and weakly prioritized.
- Browser architecture decision: export deterministic telemetry from the real
  frozen policy and replay it in a dependency-free static site. Do not ship a
  fake browser inference claim.
- Release review identified licensing, hosted URLs, and clean-clone artifact
  access as publication blockers.

### Native redesign, round one

- Rebuilt the 1440x810 renderer around a quiet slate, warm-white, and restrained
  green visual system.
- Reserved 55% of the viewport for an unobstructed road and moved detailed
  diagnostics behind the `Tab` Analysis view.
- Preserved teaching, crash, completion, collision history, policy trace, sensor,
  and traffic telemetry.
- Critic score: 7.0/10. Remaining issues were minimum text size, excess default
  labels, and two traffic telemetry fields lost in the first simplification.

### Native redesign, round two

- Increased the minimum logical font to 16 px, reduced the default drive view to
  17 functional labels, and restored traffic lane changes plus longitudinal
  acceleration in Analysis.
- Removed duplicate near-miss reporting from the crash state.
- Verified drive, Analysis, teaching, crash, and completion states at 1280x720,
  1440x810, and 1920x1080.
- Independent critic score: 10/10 with no blocking visual or usability defect.

### Browser replay, round one

- Exported three 45-second hard/dynamic routes (seeds 470000-470002) from the
  frozen Python policy with model hashes and source provenance.
- Built the static Policy Roadbook with interpolated Canvas playback, pause,
  scrub, speed, restart, route selection, responsive layout, and optional sensor
  visualization.
- Explicitly labelled the experience as recorded deterministic playback, not
  live browser inference.
- Independent critic score: 8.6/10. Requested fixes: rename one "Live telemetry"
  heading, clear the ended state when scrubbing backward, and enlarge two mobile
  controls.

### Browser replay, round two

- Renamed the section to "Replay telemetry" and standardized the experience as
  the "Frozen v1.0 layered driver" across HTML, exporter, manifest, and replay
  metadata.
- Scrubbing backward from a completed route now immediately returns the UI to
  paused playback with a correct Resume label and ARIA state.
- Restart and Sensor controls meet the 44 px mobile touch-target minimum.
- Regenerated the three replay files and their manifest hashes, then passed the
  desktop/mobile automated browser QA with no console or page errors.

### Release-candidate audit

- Python package, model staging directory, artifact manifest, model card,
  evaluation record, citation, contribution guide, security policy, CI, Pages
  workflow, release notes, and checklist prepared.
- Model and replay artifact sizes and SHA-256 hashes verified.
- Full integrated suite reached 106 passing tests; Ruff and JavaScript syntax
  checks passed.
- Independent release score: 7.6/10. Local engineering artifacts are credible;
  publication remains blocked on the user's license choice, final public URLs,
  clean staging, and hosted verification.

### Final integrated audit

- Fresh integrated verification: 106 tests passed, Ruff passed, JavaScript
  syntax passed, desktop/mobile browser QA passed, and local documentation links
  resolved.
- Model and replay artifacts matched their declared sizes and SHA-256 hashes;
  release scanning found no credentials, personal home paths, or email leaks.
- Final release staging score: 8.8/10. The remaining gates are legal and hosted:
  approve code/model licenses, publish the source tag and artifacts, deploy
  Pages, replace placeholders, and verify public clean-clone/download paths.
