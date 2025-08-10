# Changelog
All notable changes will be documented here (Keep a Changelog).

## [0.1.1] - 2025-08-09
### Added
- Initial docs pack, PR/issue templates.
- Pilot app skeleton (25 Hz VS), simulator controls.

### Fixed
- N/A

### Security
- N/A

## [0.2.0] - 2025-08-10
### Added
- Implemented full DJI bridge (registration, product connect/disconnect, flight controller acquisition).
- Added virtual stick mode configuration and body‑frame velocity control with yaw rate (angular velocity) and throttle.
- Implemented simulator start/stop with default initialization parameters.
- Added watchdog logic to `VirtualStickLoop` with 25 Hz command rate and automatic VS disable on stall.
- Added Compose-based Pilot UI: buttons for Init DJI, Enable/Disable VS, Start/Stop Simulator, status display, and sliders for manual control.
- Added training scaffolding: `train_navlnn.py` and `train_cinelnn.py` scripts using PyTorch and the provided LTCCell.

### Changed
- `VirtualStickLoop` now runs at 25 Hz (40 ms period) and includes a heartbeat watchdog.
- `MainActivity` replaced with `AppRoot` composable wiring UI to the new `DjiBridge` and `VirtualStickLoop`.
- `DjiBridge.registerAndConnect` now reads the API key from `AndroidManifest` meta‑data and manages DJI SDK callbacks.

### Fixed
- Sliders now update the global virtual stick command state and avoid restarting the sender loop unnecessarily.

### Removed
- None

### Notes
- Gradle build may fail in offline environments due to dependency downloads; ensure MSDK and Jetpack dependencies are available.

## [0.3.0] - 2025-08-10
### Added
- **Flight telemetry integration** via `FlightStateStore.startListening`.  The app now listens to the flight controller state and populates a `FlightState` with positions, body‑frame velocities, attitude, GPS fix, vision validity, obstacle distance, and a light proxy.  This state is exposed as a `StateFlow` for the UI and policy layers to consume.
- **Look‑then‑Go policy enforcement**: lateral/backwards motion is gated until a yaw scan is performed, preventing side/back collisions when obstacle sensing is valid.
- **Obstacle‑avoidance clamping**: forward velocity is automatically reduced based on time‑to‑collision computed from obstacle distance and commanded speed.  Commands that would yield a TTC < 1 s are halted.
- **Navigation feature builder**: a `buildNavFeatures` function constructs the feature vector for the NavLNN from the current `FlightState` and desired command, including computed TTC, tilt, GPS/VPS flags, obstacle distance, and light proxy.
- **Command mixer**: new `CommandMixer` object applies Look‑then‑Go, OaClamp, and NavLNN caps to produce a safe final `VirtualStickLoop.VsCommand`.  Caps are symmetric for lateral velocities and respect LNN confidence.
- **LNN inference**: `LnnPolicy` now runs TensorFlow Lite if a model is present; otherwise it returns conservative defaults.  Outputs are mapped to typed caps and clamped.
- **Policy integration in UI**: `MainActivity` now computes desired commands from sliders (vx, vy, vz, yawRate), builds features for the NavLNN, obtains outputs via `LnnPolicy`, and mixes them with the command mixer before sending via virtual stick.  Manual commands are thus subject to policy caps and OA clamping.
- **Lazy TFLite loading**: `MainActivity` attempts to load `nav_lnn.tflite` from assets at runtime.  If unavailable, the policy falls back gracefully.
- **Updated docs**: this changelog, architecture document, and agent instructions reflect the new modules and flows.

### Changed
- The `VirtualStickLoop` command provider signature now returns a `VirtualStickLoop.VsCommand` built by `CommandMixer` instead of the raw slider values.
- The pilot UI sliders now map directly to body‑frame velocities (forward/back = vx, left/right = vy, vertical = vz) and yaw rate; labels updated accordingly.
- `FlightStateStore` exposes a public `state` flow and a `startListening` method that hooks into the DJI flight controller to receive telemetry.
- `DjiBridge` starts the flight state listener when the product connects.
- `LookThenGo`, `OaClamp`, and `LnnPolicy` have documented behaviour with comments explaining their heuristics.

### Fixed
- Correctly maps body‑frame velocities to virtual stick command fields (roll = vy, pitch = vx, throttle = vz).
- Ensures forward speed is safely zeroed when time‑to‑collision is below 1 s.

### Notes
- MapGrid integration, vertical caution, classifier, and mission profiles remain TODO for future milestones.
