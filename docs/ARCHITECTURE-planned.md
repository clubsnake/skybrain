# SkyBrain — Architecture (Planned)

## Vision
Transform the original Mavic Pro into an intelligent cinematography platform with enhanced safety, autonomous navigation, and cinematic shot automation.

## Planned Additions

### Enhanced Neural Networks
- **CineLNN**: Cinema-focused neural network for:
  - Camera exposure bias optimization
  - Gimbal movement smoothing and ramps
  - Bracketing cadence for HDR captures
  - RLHF-tuned for subjective "cinematic feel"
- **Classifier**: MobileNet-lite semantic understanding (5–10 Hz):
  - Scene classification: tree/house/person/car/sky/water
  - Dynamic object detection and tracking
  - Context-aware shot selection

### Advanced Perception
- **Monocular Depth**: Real-time depth estimation (≤5 Hz)
  - Single camera depth inference
  - Integration with MapGrid for 3D understanding
  - Enhanced obstacle avoidance in complex environments
- **EKF/α-β Filter**: Extended Kalman Filter for state estimation
  - Smoother telemetry processing
  - Quality flags for gating policy decisions
  - Reduced noise in control loops

### User Experience
- **FPV Preview + HUD**: Real-time flight interface
  - Live video feed display
  - Map overlay with obstacle visualization
  - RTH (Return to Home) controls
  - Camera settings adjustment
- **Mission Profiles**: Pre-configured flight modes
  - Real-estate photography automation
  - Photogrammetry flight patterns
  - Cinematic B-roll sequences
- **Model Inspector**: Neural network transparency
  - Live visualization of LNN time constants (τ)
  - Input/output monitoring
  - Confidence scoring display

### Data Management
- **Media Ingest**: Automated content management
  - Photo/video download and organization
  - Metadata embedding with flight telemetry
  - Cloud sync integration
- **Enhanced Logging**: Comprehensive data collection
  - Flight performance analytics
  - Neural network decision logging
  - Safety event recording

## Development Milestones

### M0: Foundation (Current)
- [x] Basic Android app structure
- [x] DJI SDK integration skeleton
- [x] Virtual Stick control implementation
- [x] Safety framework (KillSwitch)
- [x] Telemetry protobuf schema
- [ ] Simulator testing and validation

### M1: Core Navigation (Next)
- [ ] Complete MapGrid obstacle integration
- [ ] Implement LookThenGo navigation policy
- [ ] OaClamp velocity limiting
- [ ] VerticalCaution rule-based system
- [ ] Basic field testing

### M2: Neural Intelligence
- [ ] NavLNN training pipeline
- [ ] Imitation learning from manual flights
- [ ] SafeDAgger implementation for safe exploration
- [ ] Model Inspector UI with τ visualization
- [ ] Confidence-based decision making

### M3: Cinematic Capabilities
- [ ] CineLNN implementation and training
- [ ] Semantic classifier integration
- [ ] Basic FPV HUD development
- [ ] Shot library implementation (Orbit, DollyReveal)
- [ ] Camera profile management

### M4: Production Features
- [ ] Monocular depth fusion
- [ ] Mission packs for different use cases
- [ ] Media ingest and management
- [ ] Advanced HUD with map overlay
- [ ] Performance optimization

## Technical Dependencies

### Hardware Requirements
- Android device (Samsung Galaxy S24U tested)
- DJI Mavic Pro (original)
- DJI RC with Mobile SDK support
- Minimum 8GB RAM for on-device neural inference

### Software Dependencies
- Android SDK 34
- Kotlin 2.0+
- TensorFlow Lite 2.14+
- DJI Mobile SDK v4.18
- Jetpack Compose
- Coroutines & StateFlow

## Risk Mitigation
- **Safety First**: All new features maintain existing safety constraints
- **Incremental Development**: Each milestone independently testable
- **Feature Flags**: New capabilities disabled by default
- **Rollback Capability**: Quick reversion to previous stable state
- **Extensive Testing**: Simulator validation before field testing

## Success Metrics
- **Safety**: Zero incidents during development and testing
- **Performance**: Maintain 25Hz Virtual Stick timing under all conditions
- **Quality**: Smooth cinematic output competitive with manual operation
- **Usability**: Non-technical users can operate advanced features
- **Reliability**: 99%+ successful flight completion rate
