#!/usr/bin/env bash
# tools/llm_context_pack.sh — bundle key docs for LLM context
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${ROOT_DIR}/llm_context.txt"
{
  echo "=== AGENT.md ==="
  cat "${ROOT_DIR}/docs/AGENT.md" 2>/dev/null || true
  echo
  echo "=== ARCHITECTURE-current.md ==="
  cat "${ROOT_DIR}/docs/ARCHITECTURE-current.md" 2>/dev/null || true
  echo
  echo "=== contracts/telemetry.md ==="
  cat "${ROOT_DIR}/docs/contracts/telemetry.md" 2>/dev/null || true
  echo
  echo "=== contracts/policy_io.md ==="
  cat "${ROOT_DIR}/docs/contracts/policy_io.md" 2>/dev/null || true
  echo
  echo "=== SAFETY.md ==="
  cat "${ROOT_DIR}/docs/SAFETY.md" 2>/dev/null || true
} > "$OUT"
echo "[OK] Wrote $OUT"
