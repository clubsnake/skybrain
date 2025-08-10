# Contributing to SkyBrain

Thank you for your interest in contributing to SkyBrain! This document provides comprehensive guidelines for contributing to this safety-critical drone automation project.

## 🚀 Quick Start

### Prerequisites
- Android Studio with Kotlin 2.0+
- DJI Mavic Pro (original) for testing
- Samsung Galaxy S24U (recommended) or compatible Android device
- Git with LFS for model files

### Setup
```bash
# Clone the repository
git clone https://github.com/armand/skybrain.git
cd skybrain

# Install git hooks for quality control
git config core.hooksPath .githooks
chmod +x .githooks/pre-commit

# Run quality checks
./tools/check.sh

# Open in Android Studio and sync Gradle
```

## 📋 Before You Start

### Required Reading
1. Read `docs/AGENT.md` - **MANDATORY** for all contributors
2. Review `docs/ARCHITECTURE-current.md` for system understanding
3. Study `docs/SAFETY.md` for safety requirements
4. Check `docs/naming.md` for naming conventions

## 🛠️ Development Workflow

### Branch Naming
```
feature/add-orbital-shots
fix/virtual-stick-timeout
docs/update-architecture
safety/enhance-killswitch
```

### Commit Message Format
Use conventional commits:
```
feat(policy): add vertical obstacle avoidance
fix(sdk): resolve virtual stick timeout issue
docs(arch): update module documentation
safety(kill): enhance emergency stop reliability
```

### Pull Request Requirements
- [ ] All quality checks pass (`./tools/check.sh`)
- [ ] Unit tests added/updated
- [ ] Documentation updated (if interfaces changed)
- [ ] Safety impact assessed
- [ ] Simulator testing completed
- [ ] ADR created (for architectural changes)

## 🔍 Code Quality Standards

### Automated Checks
Every commit is automatically checked for:
- **ktlint**: Kotlin code formatting
- **detekt**: Static analysis and code quality
- **Naming conventions**: Prevent LLM-generated errors
- **Package structure**: Maintain architecture boundaries
- **Safety validation**: No TODOs in critical files

### Style Guidelines
- **Kotlin**: Idiomatic, coroutines, StateFlow/SharedFlow
- **Python**: black, isort, type hints
- **Protobuf**: Explicit fields; never reuse numeric tags

## 🛡️ Safety-First Development

### Safety-Critical Files (Require TWO Approvals)
- `app/src/main/java/com/armand/skybrain/safety/KillSwitch.kt`
- `app/src/main/java/com/armand/skybrain/sdk/VirtualStickLoop.kt`
- `app/src/main/java/com/armand/skybrain/sdk/DjiBridge.kt`
- `docs/SAFETY.md`

### Safety Rules (NON-NEGOTIABLE)
1. **Never disable safety systems** without explicit ADR approval
2. **Always test in simulator first** - minimum 2 minutes without watchdog trips
3. **Respect flight boundaries** - maintain geofence and altitude limits
4. **Preserve emergency stop** - KillSwitch must always be functional

### Testing Protocol
1. **Simulator Testing**: DJI Simulator for 5+ minutes minimum
2. **Quality Gates**: `./tools/check.sh` must pass
3. **Field Testing**: Open area, LOS, low speed (≤1.5 m/s)

## 📚 Documentation Requirements

### Every PR Must Update
- `docs/contracts/*` if interfaces change
- `docs/ARCHITECTURE-*` if module structure changes
- `docs/CHANGELOG.md` for user-facing changes
- ADR for architectural decisions

### Living Documentation
- Architecture docs auto-sync with code
- Contract files are source of truth
- Model cards track neural network assumptions

## 🧪 Testing Requirements

### Required Tests
- **Unit Tests**: 80% coverage for business logic
- **Simulator Tests**: Flight behavior validation
- **Safety Tests**: All safety systems verified
- **Integration Tests**: Module interactions

## 🚨 Safety Protocol

### If Safety Issue Found
1. Create critical issue with "SAFETY" label
2. Halt related development immediately
3. Implement fix with safety validation
4. Update safety documentation

## 🏗️ Architecture Guidelines

### Module Boundaries
- **SDK**: DJI integration only
- **Safety**: Isolated and reliable
- **Perception**: Sensor processing
- **Policy**: Decision making within safety bounds
- **UI**: Presentation layer only

### Neural Network Standards
- Model size: <10MB for mobile
- Inference latency: <200ms target
- Always provide confidence scores
- Models suggest, safety systems decide

---

**Remember: Safety is our top priority. When in doubt, ask questions and test thoroughly.**
