package com.armand.skybrain.policy

import com.armand.skybrain.sdk.FlightState
import kotlin.math.abs
import kotlin.math.sqrt

private const val TTC_INF_PLACEHOLDER = 9999f
private const val MIN_VX_FOR_TTC = 1e-3f

/**
 * Builds the input feature vector for the navigation LNN (NavLNN) based on
 * the current flight state and the desired command. The feature order is
 * defined in `docs/contracts/policy_io.md`.
 */
fun buildNavFeatures(
    state: FlightState,
    desired: LookThenGo.Desired,
): FloatArray {
    // Time to collision estimate: positive vx and valid OA distance => dist / vx
    val ttc =
        if (state.oaValid && desired.vx > MIN_VX_FOR_TTC) {
            state.oaDistM / desired.vx
        } else {
            Float.POSITIVE_INFINITY
        }
    val tilt = abs(state.pitchDeg)

    // Placeholders for future integrations
    val windMag = 0f
    val windDirRel = 0f
    val costAhead = 0f
    val costLeft = 0f
    val costRight = 0f
    val heightHeadroom = 0f
    val missionProgress = 0f
    val desiredSpeed = sqrt(desired.vx * desired.vx + desired.vy * desired.vy)

    return floatArrayOf(
        // Body-frame velocities
        desired.vx,
        desired.vy,
        desired.vz,
        desired.yawRate,
        // Orientation
        state.yawDeg,
        tilt,
        // GPS fix
        state.gpsFix.toFloat(),
        // VPS valid flag
        if (state.vpsValid) 1f else 0f,
        // Obstacle avoidance distance
        state.oaDistM,
        // Time to collision
        if (ttc.isFinite()) ttc else TTC_INF_PLACEHOLDER,
        // Wind estimates
        windMag,
        windDirRel,
        // Light proxy
        state.lightProxy,
        // Map costs
        costAhead,
        costLeft,
        costRight,
        // Height headroom
        heightHeadroom,
        // Mission progress
        missionProgress,
        // Desired speed
        desiredSpeed,
    )
}
