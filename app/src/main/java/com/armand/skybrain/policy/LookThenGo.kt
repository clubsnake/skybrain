package com.armand.skybrain.policy

import com.armand.skybrain.sdk.FlightState

object LookThenGo {
    data class Desired(val vx: Float, val vy: Float, val vz: Float, val yawRate: Float)

    private const val LATERAL_THRESHOLD_MS = 0.1f
    private const val BACKWARDS_EPSILON_MS = 0.01f

    /**
     * Enforces a "look-then-go" policy.  Before allowing significant
     * lateral (vy) or backwards (negative vx) motion, the aircraft must
     * yaw-align such that the forward obstacle sensors (and camera) face
     * the desired direction of travel.  If a yaw adjustment is required,
     * this function temporarily zeroes the translational velocities and
     * leaves the yaw rate unchanged.  Once aligned, the caller should
     * resume sending the original desired velocities.
     *
     * The policy uses very simple heuristics:
     *   • If vy magnitude exceeds a small threshold (0.1 m/s) or vx is
     *     negative (flying backwards), and obstacle sensing is valid, then
     *     translational motion is suppressed until yaw is corrected.
     *   • Yaw alignment is left to the caller via a non-zero desired yawRate.
     *   • Vertical and yaw motion are never modified here.
     *
     * @param state The current flight state (yaw, obstacle validity, etc.)
     * @param desired The desired body‑frame velocities (m/s) and yaw rate (deg/s)
     * @return The potentially modified desired command
     */
    fun enforce(
        state: FlightState,
        desired: Desired,
    ): Desired {
        // Only apply the look-then-go gate when obstacle avoidance is valid.
        // If no OA sensors are active (e.g. in Sport mode) we don't force a yaw scan.
        if (state.oaValid) {
            val lateral = kotlin.math.abs(desired.vy)
            val backwards = desired.vx < -BACKWARDS_EPSILON_MS
            if (lateral > LATERAL_THRESHOLD_MS || backwards) {
                // Suppress translational velocities; allow yaw and vertical
                return desired.copy(vx = 0f, vy = 0f)
            }
        }
        return desired
    }
}
