# SkyBrain — Architecture (Current)

## Purpose
Safety-first Android Kotlin app that layers intelligent policy + Liquid Neural Networks (LNNs) on top of DJI stabilization via Virtual Stick for the original Mavic Pro.

## Core Principles
1. **Safety First**: Never bypass DJI stabilization; add intelligence on top
2. **LNN-Driven**: Use Liquid Time-Constant networks for adaptive, real-time decisions
3. **Modular Design**: Clear separation of concerns with well-defined interfaces
4. **Transparent**: User can see and understand neural network decisions

## Module Architecture

### SDK Layer (`com.armand.skybrain.sdk`)
    - **DjiBridge.kt**: Core DJI Mobile SDK v4.x integration
      - Status management (Disconnected/Connecting/Connected)
      - Virtual Stick mode enablement/disablement
      - Body‑frame velocity commands (roll, pitch, yaw_rate, throttle)
      - Starts the flight state listener upon connection
    - **VirtualStickLoop.kt**: 25 Hz command sender with heartbeat watchdog
      - Coroutine-based sender loop; stops on exceptions or >200 ms latency
      - Command provider abstraction
    - **FlightStateStore.kt**: Telemetry processing
      - Provides a `StateFlow<FlightState>` with lat/lon/alt, body‑frame velocities, attitude, GPS fix, vision validity, obstacle distance, and light proxy
      - Hooks into the DJI flight controller via `startListening`

### Safety Layer (`com.armand.skybrain.safety`)
- **KillSwitch.kt**: Emergency stop mechanism
  - Immediate VS loop termination
  - VS mode disabling
  - Fail-safe engagement

### Perception Layer (`com.armand.skybrain.perception`)
- **MapGrid.kt**: Occupancy grid mapping
  - 3D grid (120x120x60m @ 1m resolution)
  - Probabilistic occupancy (p_occ, sigma, age)
  - Cost inflation for safe navigation

### Policy Layer (`com.armand.skybrain.policy`)
    - **LnnPolicy.kt**: Neural network inference
      - TensorFlow Lite wrapper (gracefully handles null interpreter)
      - Maps NavLNN raw outputs to typed caps (v_cap_fwd, v_cap_lat, etc.)
      - Returns conservative defaults if no model present
    - **LookThenGo.kt**: Look‑first navigation policy
      - Blocks lateral/back motion until yaw scan is complete when OA is valid
    - **OaClamp.kt**: Obstacle avoidance velocity clamping
      - Reduces forward speed based on time‑to‑collision; hard stop if TTC < 1 s
    - **CommandMixer.kt**: Safety & policy combiner
      - Applies Look‑then‑Go, OaClamp, and NavLNN caps to produce final stick commands
    - **FeatureBuilder.kt**: NavLNN feature construction
      - Converts `FlightState` and desired command into the input feature vector defined in `policy_io.md`

### UI Layer (`com.armand.skybrain.ui`)
    - **MainActivity.kt**: Jetpack Compose interface
      - DJI initialization controls, VS enable/disable, simulator controls
      - Status display and watchdog toasts
      - Sliders for forward/back (vx), left/right (vy), vertical (vz), and yaw rate
      - Calls into the policy layer to build NavLNN features, run inference, mix commands, and send via VS

### Training Infrastructure (`training/`)
- **pytorch/ltc_cell.py**: Liquid Time-Constant cell implementation
  - Adaptive time constants (tau)
  - TFLite exportable
  - Variable memory behavior

## Data Flow Architecture

```
DJI Telemetry → FlightStateStore → Nav Feature Builder → LNN Inference
                      ↓                                 ↓
               MapGrid Update (future)            LookThenGo
                      ↓                                 ↓
               VerticalCaution (TODO) ← OaClamp ← CommandMixer → Virtual Stick
                                                     ↓
                                                 DjiBridge → Aircraft
```

## Timing Specifications
- **Virtual Stick Sender**: 25 Hz (40ms period)
- **Policy Computation**: 10-20 Hz adaptive
- **Neural Network Inference**: ≤5 Hz for complex models
- **Watchdog Timeout**: 200ms maximum latency

## Safety Envelope
1. **Always test in simulator first** (props off)
2. **Dual watchdogs**: inbound heartbeat, outbound braking detector
3. **Speed limits**: ≤1.5 m/s lateral, ≤8° tilt for initial flights
4. **KillSwitch**: immediate stop capability
5. **LOS requirement**: line-of-sight, open field only

## Communication Interfaces
- **Telemetry**: See `contracts/telemetry.md` (protobuf schema)
- **Policy I/O**: See `contracts/policy_io.md` (NavLNN interface)
- **Neural Network**: TensorFlow Lite format

## Deployment Target
- **Platform**: Android (minSdk 26, targetSdk 34)
- **Hardware**: Samsung Galaxy S24U (primary test device)
- **Aircraft**: DJI Mavic Pro (original) via Mobile SDK v4.18
- **Package**: `com.armand.skybrain`

## Version Control
- **Current Version**: 0.1.0
- **Schema Version**: 1 (telemetry protobuf)
- **Kotlin**: 2.0.0
- **Compose**: 1.5.14
- **TensorFlow Lite**: 2.14.0
