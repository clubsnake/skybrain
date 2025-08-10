# Naming Conventions

## Purpose
Establish consistent naming patterns to prevent LLM-generated code errors, improve maintainability, and ensure clear separation of concerns.

## Package Structure
```
com.armand.skybrain/
├── sdk/          # DJI integration and hardware communication
├── safety/       # Safety systems and kill switches
├── perception/   # Sensor processing and world modeling
├── policy/       # Decision making and neural networks
├── ui/           # User interface and Jetpack Compose
├── capture/      # Camera and media functionality
├── learning/     # Data collection and training support
└── proto/        # Generated protobuf classes
```

## Class Naming

### Core Classes (Never Rename Without ADR)
- `DjiBridge` - DJI SDK integration
- `VirtualStickLoop` - Command sender loop
- `FlightStateManager` - State estimation
- `KillSwitch` - Emergency stop
- `MapGrid` - Occupancy mapping
- `LnnPolicy` - Neural network policy

### Naming Patterns
- **Classes**: `PascalCase` (e.g., `FlightStateManager`)
- **Interfaces**: `PascalCase` with descriptive names (e.g., `PolicyProvider`)
- **Data classes**: `PascalCase` with meaningful names (e.g., `FlightTelemetry`)
- **Enums**: `PascalCase` for type, `UPPER_SNAKE_CASE` for values

```kotlin
enum class FlightMode {
    MANUAL,
    AUTONOMOUS,
    EMERGENCY_RTH
}
```

## Function and Variable Naming

### Functions
- **Public API**: `camelCase` with action verbs (e.g., `enableVirtualStick()`)
- **Private methods**: `camelCase` starting with verb (e.g., `processTelemtry()`)
- **Suspend functions**: Include "suspend" context when ambiguous (e.g., `suspendAndWait()`)

### Variables
- **Properties**: `camelCase` (e.g., `batteryLevel`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `MAX_VELOCITY_MS`)
- **Private fields**: Leading underscore optional for backing properties (e.g., `_status`)

### Common Abbreviations (Approved)
- `vs` → Virtual Stick
- `oa` → Obstacle Avoidance  
- `lnn` → Liquid Neural Network
- `rtl` → Return to Launch
- `gps` → Global Positioning System
- `vps` → Visual Positioning System
- `ekf` → Extended Kalman Filter

## File Naming

### Kotlin Files
- Match primary class name exactly: `DjiBridge.kt`
- Utilities: descriptive names ending in purpose (e.g., `MathUtils.kt`)
- Extensions: `ClassNameExtensions.kt` pattern

### Resource Files
- Layouts: `activity_main.xml`, `fragment_pilot.xml`
- Strings: descriptive keys with module prefix (e.g., `pilot_button_init_dji`)

## Protobuf Conventions
- **Message names**: `PascalCase` (e.g., `FlightTelemetry`)
- **Field names**: `snake_case` (e.g., `timestamp_ns`)
- **Never renumber tags**: Only add new fields with new numbers
- **Required versioning**: Include `schema_version` in all messages

```protobuf
message FlightTelemetry {
  int32 schema_version = 1;
  fixed64 timestamp_ns = 2;
  float altitude_m = 3;
  // ... more fields
}
```

## Neural Network Naming

### Model Types
- `NavLNN` - Navigation Liquid Neural Network
- `CineLNN` - Cinematography Liquid Neural Network
- Models use `PascalCase` with descriptive suffixes

### Model Files
- Training: `nav_lnn_v1.py`
- Exported: `nav_lnn_v1.tflite`
- Version in filename for tracking

## Database and Storage
- **Tables**: `snake_case` (e.g., `flight_sessions`)
- **Columns**: `snake_case` (e.g., `created_at`)
- **JSON keys**: `snake_case` for consistency with protobuf

## Version Control
- **Branches**: `feature/description` or `fix/issue-description`
- **Tags**: Semantic versioning `v0.1.0`
- **Commit messages**: Conventional commits format

## Anti-Patterns (Forbidden)

### Never Use These Names
- `manager` suffix without context (too generic)
- `helper` or `util` without specific purpose
- `data` as class prefix (redundant with data class)
- Single letter variables (except loop counters)
- Abbreviations not in approved list

### LLM-Specific Guidelines
- **No generic names**: Avoid `Process`, `Handle`, `Manage` without context
- **Be specific**: `VelocityCommand` not `Command`
- **Avoid overloading**: Don't reuse names across different modules
- **Include units**: `altitudeM` not `altitude`

## Migration Guidelines
When renaming existing code:
1. Create ADR documenting the change
2. Update all documentation simultaneously
3. Use IDE refactoring tools to ensure consistency
4. Update tests and comments
5. Verify no broken references remain

## Validation Tools
- ktlint: Enforce Kotlin style guide
- detekt: Static analysis for code quality
- Custom rules: Package structure validation
- Pre-commit hooks: Prevent naming violations

## Examples

### Good Naming
```kotlin
class FlightStateManager {
    private val _telemetry = MutableStateFlow<FlightTelemetry?>(null)
    val telemetry: StateFlow<FlightTelemetry?> = _telemetry
    
    suspend fun updateFromDji(rawData: DjiTelemetryRaw) {
        val processed = processRawTelemetry(rawData)
        _telemetry.value = processed
    }
}
```

### Bad Naming (Don't Do This)
```kotlin
class Manager {  // Too generic
    var data: Flow<Any>?  // Unclear type
    fun process(x: Any)   // No context
}
```
