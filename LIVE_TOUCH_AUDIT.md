# Live touch update — September 20, 2026

## Run the dashboard

From the Skinless folder in VS Code PowerShell:

```powershell
.venv\Scripts\Activate.ps1
python -m ml.ui_server --serial-port COM5
```

The live activity threshold defaults to `1.5` gain-corrected ADC-RMS above the
powered-on idle envelope. If genuine very light touches still do not register,
try `--threshold 0.75`. If the activity bars or heat fire while untouched, use
`--threshold 3`. Always restart with the robot untouched during startup.

Open http://localhost:8080 and refresh after updates. Keep the powered-on robot
untouched for the first two seconds. Close Arduino Serial Monitor and any other
collector first. Only one process can own COM5. Ctrl+C in the server terminal
stops it; do not launch a second copy while it is running.

No full gesture retraining is needed for this update. Existing gesture models
are still used. The motion artifact was retrained using the current gain scaling.

## What the audit actually found

| Issue | Evidence and change |
|---|---|
| Delay | Serial reading, spatial forest inference and HTTP polling could block/delay delivery. USB reads now consume chunks; a dedicated acquisition thread maintains a newest-only ring. Persistent server-sent events deliver current state, not queued frames. Gesture and temporal-forest inference run separately from live heat. |
| Inconsistent localization features | Spatial training used 32-sample RMS features, whereas live features used a shorter window. Live localization now uses 32 samples and the motion history uses the training stride of 10 samples. |
| Sensitivity bias | Uncontrolled-force recordings gave GPIO15 a normalization scale near 0.47, boosting it more than twofold. Gain compensation is now conservative (square-root scaling, bounded 0.75–1.5). These are estimates, not measured sensor sensitivities. |
| Weak/intermittent touches | An extra amplitude gate and high intensity floor suppressed weak contacts. The learned fast presence gate remains; strong onsets can bypass its 8 ms debounce. A 70 ms release interval bridges brief gaps without retaining a long visible touch. |
| Snapping/interpolation | Squared channel weights and strong side/motion priors overrode intermediate locations. Linear energy weights, weaker priors and explicit isolated-channel anchors preserve intermediate coordinates. |
| Cross-body heat | A Euclidean heat sphere could illuminate the opposite shell. The shader now also limits heat by body-surface angle. Opposite-side jumps clear old trails. This fixes rendering leakage, not incorrectly inferred sides. |
| Idle heat | The recorded no-touch-trained presence gate and robust startup/adaptive noise estimate remain active. Replay still shows false positives; they are not eliminated. |
| Orbit/contrast | Camera-up previously changed after orbit controls initialized. Both now use the conventional Y-up basis and normal drag direction. Heat is red-to-white against a subdued cyan shell, with strength-dependent brightness and frame-by-frame 180 ms decay. |

The active-event side tracker now locks immediately to left/right/top evidence.
Left estimates are constrained to the left surface hemisphere and right estimates
to the right hemisphere. Six consecutive contrary updates are required before a
real around-body crossing is accepted. The renderer uses a 24-point interpolated
trail with a 580 ms maximum lifetime and exponential fade, while contact onset
remains immediate. The **Demo stroke** button previews the left-side trail.

The actual stream measured about **665–677 samples/second**, not the configured
nominal 1000. Twelve samples span about 18 ms; 32 span about 48 ms. These are
trailing windows, not additional windows collected after a trigger. Latest-state
processing was roughly 4–6 ms at the 95th percentile during observed live checks.
Those numbers **are not physical-touch-to-screen latency measurements**: firmware,
USB buffering, event detection, display timing and rendering still contribute.
Zero latency is not possible and a synchronized physical test remains necessary.

## Important limitation in the existing recordings

Across 700 five-channel real recordings:

- GPIO6 / GPIO7 correlation: **0.998**; 94.9% of their sample values match exactly
  (including zero samples).
- GPIO5 / GPIO15 correlation: **0.992**.

That makes the top-rear/front-right pair and rear-right/front-left pair difficult
to distinguish from these recordings. Software cannot reliably recover position
when different locations produce indistinguishable input. This is evidence of
ambiguity, not proof of a particular wiring fault. Check the labelled pin routing,
separate signal connections and sensor mounting, and compare an isolated tap on
6 versus 7, then 5 versus 15. The short collector below prints their responses and
warns when these channels closely duplicate each other.

Piezo vibration amplitude is a **relative intensity proxy**, not calibrated force;
multi-channel participation is not a measurement of physical hand/contact area.
Static pressure may produce little vibration. Five sensors do not guarantee an
arbitrary continuous trajectory, multiple simultaneous contacts or whole-body
coverage. Surface geometry remains the existing approximate torso parameterization
projected onto the GLB, not a measured geodesic map.

## Minimum useful extra data: 16 short recordings

The old recordings label sensor regions; motion paths label direction but do not
measure finger position at each instant. The new collector adds exact known-point
responses rather than requesting another large gesture dataset.

Stop the dashboard with Ctrl+C, then:

```powershell
python -m ml.collect_surface --port COM5 --count 2
```

It prompts for the five existing sensor locations and three additional points:

| Location | What to touch |
|---|---|
| rear_left_side | Directly above GPIO4 |
| rear_right_side | Directly above GPIO5 |
| rear_top_butt | Directly above GPIO6, on top near the rear |
| front_right_side | Directly above GPIO7 |
| front_left_side | Directly above GPIO15 |
| left_middle | Halfway along the side between GPIO15 and GPIO4 |
| right_middle | Halfway along the side between GPIO7 and GPIO5 |
| top_middle | Top centerline at the same front/rear midpoint as the side midpoints |

For each prompt: remove your hand, press Enter, **wait for GO**, then touch.
Trial 1: one light tap. Trial 2: a small gentle rub centered on the same point.
Keep the robot powered on in the usual stance and use comparable strength across
points. Do not perform a long stroke for these stationary coordinate labels.
Each capture lasts about 1.3 seconds at the observed board rate. The collector
waits for an above-noise onset; use a clear gentle tap if a motionless touch does
not trigger. A complete session is 8 points × 2 trials = 16 recordings.

Files append under `data/surface_calibration`; Ctrl+C preserves saved trials.
Rerunning appends, not overwrites. Complete all eight points for useful coverage;
five non-collinear recorded coordinates are required to enable the response field.
The saved point coordinates are normalized assumptions: place midpoint marks as
described. Exact mounting changes require updating the body map as well.

Restart the same dashboard command. It automatically fits and loads a triangulated
response field from these recordings: interpolation happens in measured signal
space over known surface coordinates. No separate training command is needed.
Existing directional recordings remain a weak background prior; live evidence
and measured-point mapping take precedence. Ambiguous channels still need fixing.

The loader now validates the response field against its recorded known points.
If median normalized surface error exceeds 0.16, or more than 25% of recordings
exceed 0.30, it is automatically rejected and the gain-corrected physical centroid
is used. This prevents a noisy calibration from making localization worse.

## Verification and honest results

```powershell
python -m unittest ml.test_live_surface
python -m ml.validate_surface
```

Five regression tests pass: chunked serial parsing/reset, newest-only buffering,
all five physical anchors and both interpolation directions, and measured-point
interpolation, and continued heat estimates while the temporal model is deliberately
blocked. Synthetic equal corrected front/rear input gives u≈0.54 on either
side, rather than snapping to u=0.24 or u=0.80. GPIO6 alone maps to (0.82, 0.02).
These are software tests, not proof that real independent channels are available.

The replay report is saved to `data/surface_validation.json`. It uses existing
training data and a three-frame approximation to debounce; it is **not held-out
physical accuracy or an exact replay of wall-clock live operation**. With the
high-confidence onset shortcut, recording-level detection ranges from 49% to 72%
across locations; 0.93% of recorded idle frames pass this replay gate (previous
three-frame rule: 0.73%). This trade-off improves onset response but does not solve
all missed/false touches. Motion holdout side accuracy is 80%, with normalized-u
MAE 0.092 against assumed constant-speed labels. The reused split and inferred
positions do not establish centimeter accuracy or accurate finger tracking.

For a short physical acceptance check after collecting: leave untouched for 10 s;
tap each sensor; touch each midpoint; then sweep left front→rear and rear→front,
and repeat on the right. Confirm correct side, order, midpoint and prompt release.
Do not collect more long path datasets unless that check identifies a specific gap.

## Main code

- `ml/live_dashboard.py`, `ml/live_stream.py`: independent acquisition, heat and classification; telemetry and streaming.
- `ml/source_serial.py`: chunked reads and complete input-buffer reset.
- `ml/spatial_tracker.py`: conservative gains, interpolation, asynchronous weak motion prior.
- `ml/surface_calibration.py`, `ml/collect_surface.py`: small measured response-field workflow.
- `ml/test_live_surface.py`, `ml/validate_surface.py`: repeatable regression/audit checks.
- `ui/dist/app.js`: surface-limited heat, conventional orbit, contrast and streaming.

The original waveform interface, gesture collection/training, raw data and 3D
dashboard remain available. No original recordings were deleted or relabelled.
