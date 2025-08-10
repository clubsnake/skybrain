package com.armand.skybrain.policy

import com.armand.skybrain.sdk.FlightState
import com.armand.skybrain.sdk.VirtualStickLoop

/**
 * Combines the user/mission desired command with safety policies and neural
 * network caps.  The mixer applies (in order):
 *  1. Look‑then‑Go gate: blocks lateral/back motion unless yaw scan is done.
 *  2. Obstacle‑avoidance clamp: slows/halts forward velocity based on OA distance.
 *  3. NavLNN caps: saturates each velocity component to the liquid network’s
 *     recommended maximums; optionally modulates yaw rate and ramp gain.
 *
 * This class is stateless; callers should retain the latest [LnnPolicy.Output]
 * from the model and pass it along with the current [FlightState] and the
 * desired command.
 */
object CommandMixer {
    /**
     * Mixes a desired [LookThenGo.Desired] command with safety and LNN caps.
     *
     * @param state Current flight state (positions, velocities, sensor validity)
     * @param desired Base desired body‑frame velocities and yaw rate
     * @param nnOut Latest LNN output; if null, defaults are used
     * @return A [VirtualStickLoop.VsCommand] ready to send via Virtual Stick
     */
    fun mix(
        state: FlightState,
        desired: LookThenGo.Desired,
        nnOut: LnnPolicy.Output?,
    ): VirtualStickLoop.VsCommand {
        // First apply look‑then‑go gate (yaw-align before lateral/back moves)
        var gated = LookThenGo.enforce(state, desired)
        // Then apply OA clamping on forward velocity
        val oaCmd = OaClamp.apply(OaClamp.Cmd(gated.vx, gated.vy, gated.vz, gated.yawRate), state.oaDistM)
        // Apply NavLNN caps if provided
        if (nnOut != null) {
            // Cap forward velocity between -vCapLat (allow reverse slower) and vCapFwd
            val vx = oaCmd.vx.coerceIn(-nnOut.vCapLat, nnOut.vCapFwd)
            // Lateral velocity symmetric cap
            val vy = oaCmd.vy.coerceIn(-nnOut.vCapLat, nnOut.vCapLat)
            // Vertical velocity cap
            val vz = oaCmd.vz.coerceIn(-nnOut.vCapVert, nnOut.vCapVert)
            // Yaw rate cap
            val yaw = oaCmd.yawRate.coerceIn(-nnOut.yawRateCap, nnOut.yawRateCap)
            gated = LookThenGo.Desired(vx, vy, vz, yaw)
        } else {
            gated = LookThenGo.Desired(oaCmd.vx, oaCmd.vy, oaCmd.vz, oaCmd.yawRate)
        }
        // Map body-frame velocities to VirtualStickLoop command fields:
        // roll = lateral velocity (vy), pitch = forward velocity (vx), throttle = vertical velocity (vz)
        return VirtualStickLoop.VsCommand(
            roll = gated.vy,
            pitch = gated.vx,
            yawRate = gated.yawRate,
            throttle = gated.vz,
        )
    }
}
