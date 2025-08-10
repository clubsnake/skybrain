#!/usr/bin/env bash
# SkyBrain Quality Control Script
# Runs all static analysis and validation checks

set -euo pipefail

echo "🔍 SkyBrain Quality Control"
echo "=========================="

# Check if we're in the right directory
if [ ! -f "settings.gradle.kts" ]; then
    echo "❌ Error: Run this script from the project root directory"
    exit 1
fi

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check for required tools
echo "📋 Checking prerequisites..."
if ! command_exists ./gradlew; then
    echo "❌ Error: Gradle wrapper not found"
    exit 1
fi

# Run ktlint formatting check
echo "🎨 Running ktlint format check..."
./gradlew ktlintCheck || {
    echo "❌ ktlint format issues found. Run './gradlew ktlintFormat' to fix."
    exit 1
}

# Run detekt static analysis
echo "🔍 Running detekt static analysis..."
./gradlew detekt || {
    echo "❌ detekt found issues. Check reports in app/build/reports/detekt/"
    exit 1
}

# Validate protobuf schema
echo "📄 Validating protobuf schemas..."
./gradlew :app:generateDebugProto :app:generateReleaseProto || {
    echo "❌ Protobuf generation failed"
    exit 1
}

# Check naming conventions
echo "📝 Checking naming conventions..."
find app/src/main/java -name "*.kt" | while read -r file; do
    # Check for forbidden patterns
    if grep -q "class.*Manager[^a-zA-Z]" "$file"; then
        echo "❌ Generic 'Manager' class found in $file - use specific naming"
        exit 1
    fi
    if grep -q "class.*Helper[^a-zA-Z]" "$file"; then
        echo "❌ Generic 'Helper' class found in $file - use specific naming"
        exit 1
    fi
    if grep -q "class.*Util[^a-zA-Z]" "$file"; then
        echo "❌ Generic 'Util' class found in $file - use specific naming"
        exit 1
    fi
done

# Validate package structure
echo "🏗️ Validating package structure..."
expected_packages=(
    "com/armand/skybrain/sdk"
    "com/armand/skybrain/safety"
    "com/armand/skybrain/perception"
    "com/armand/skybrain/policy"
    "com/armand/skybrain/ui"
)

for package in "${expected_packages[@]}"; do
    if [ ! -d "app/src/main/java/$package" ]; then
        echo "⚠️ Warning: Expected package directory not found: $package"
    fi
done

# Check for TODOs in critical safety files
echo "🛡️ Checking safety-critical files for TODOs..."
safety_files=(
    "app/src/main/java/com/armand/skybrain/safety/KillSwitch.kt"
    "app/src/main/java/com/armand/skybrain/sdk/VirtualStickLoop.kt"
)

for file in "${safety_files[@]}"; do
    if [ -f "$file" ] && grep -q "TODO" "$file"; then
        echo "❌ Safety-critical file $file contains TODO items"
        grep -n "TODO" "$file"
        exit 1
    fi
done

# Validate documentation sync
echo "📚 Checking documentation consistency..."
if [ -f "docs/ARCHITECTURE-current.md" ]; then
    # Check that all modules mentioned in docs exist
    while IFS= read -r line; do
        if [[ $line =~ \*\*([^*]+)\.kt\*\* ]]; then
            class_name="${BASH_REMATCH[1]}"
            if ! find app/src/main/java -name "${class_name}.kt" -type f | grep -q .; then
                echo "⚠️ Warning: Documentation mentions $class_name.kt but file not found"
            fi
        fi
    done < "docs/ARCHITECTURE-current.md"
fi

# Run Android lint
echo "🔧 Running Android lint..."
./gradlew lint || { echo "❌ Android Lint failed"; exit 1; }

echo "✅ All quality checks passed!"
echo ""
echo "🚀 Ready for development or commit"
echo "   To run individual checks:"
echo "   • ./gradlew ktlintCheck    - Code formatting"
echo "   • ./gradlew detekt         - Static analysis" 
echo "   • ./gradlew qualityCheck   - All checks combined"
