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
  - Virtual Stick mode enablement
  - Body-frame velocity commands (roll, pitch, yaw_rate, throttle)
- **VirtualStickLoop.kt**: 25 Hz command sender with watchdog
  - Coroutine-based sender loop
  - Watchdog protection (200ms timeout)
  - Command provider abstraction
- **FlightStateManager.kt**: State estimation and telemetry processing

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
  - TensorFlow Lite wrapper
  - NavLNN input/output mapping
  - Confidence scoring
- **LookThenGo.kt**: Look-first navigation policy
- **OaClamp.kt**: Obstacle avoidance velocity clamping

### UI Layer (`com.armand.skybrain.ui`)
- **MainActivity.kt**: Jetpack Compose interface
  - DJI initialization controls
  - Simulator controls
  - Virtual Stick sliders for manual control

### Training Infrastructure (`training/`)
- **pytorch/ltc_cell.py**: Liquid Time-Constant cell implementation
  - Adaptive time constants (tau)
  - TFLite exportable
  - Variable memory behavior

## Data Flow Architecture

```
DJI Telemetry → FlightStateManager → Policy Layer → Virtual Stick Commands
                      ↓                    ↓
                 MapGrid Update     LNN Inference
                      ↓                    ↓
                Safety Validation → Command Mixing → DjiBridge → Aircraft
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
