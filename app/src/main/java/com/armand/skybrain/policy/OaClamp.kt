package com.armand.skybrain.policy

object OaClamp {
    data class Cmd(val vx: Float, val vy: Float, val vz: Float, val yawRate: Float)

    private const val MIN_TTC_SECONDS = 3f
    private const val HARD_TTC_SECONDS = 1f

    /**
     * Applies obstacle‑avoidance clamping to the desired body‑frame command.
     *
     * Only the forward velocity (vx) is modified.  When flying forwards
     * (positive vx) and a valid obstacle distance is provided, the function
     * computes a safe maximum velocity based on a minimum desired time‑to‑collision
     * (TTC).  If the commanded velocity would yield a TTC below the minimum,
     * it is reduced proportionally.  If the TTC is below a hard floor (1 s),
     * the forward velocity is set to zero to perform a stop/hover.
     *
     * For lateral (vy), vertical (vz), and yaw rates, the command is passed
     * through unchanged.  Additional clamping (e.g. vertical caution) is
     * handled elsewhere.
     *
     * @param cmd Desired body‑frame velocities
     * @param oaDistM Forward obstacle distance in metres (Float.POSITIVE_INFINITY if none)
     * @return A command with clamped forward speed
     */
    fun apply(
        cmd: Cmd,
        oaDistM: Float,
    ): Cmd {
        var vx = cmd.vx
        // Only clamp if moving forward and we have a finite OA distance
        if (vx > 0f && oaDistM.isFinite()) {
            // TTC thresholds
            val ttc = oaDistM / vx
            when {
                ttc < HARD_TTC_SECONDS -> vx = 0f
                ttc < MIN_TTC_SECONDS -> {
                    // Scale down vx so that TTC equals minTtc
                    val allowedVx = oaDistM / MIN_TTC_SECONDS
                    vx = kotlin.math.min(vx, allowedVx)
                }
                // else: keep vx unchanged
            }
        }
        return Cmd(vx, cmd.vy, cmd.vz, cmd.yawRate)
    }
}
