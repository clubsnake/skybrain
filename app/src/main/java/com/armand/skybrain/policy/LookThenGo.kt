package com.armand.skybrain.policy

import com.armand.skybrain.sdk.FlightState

object LookThenGo {
    data class Desired(val vx: Float, val vy: Float, val vz: Float, val yawRate: Float)

    fun enforce(
        state: FlightState,
        desired: Desired,
    ): Desired {
        // TODO: yaw-align before lateral/back moves
        return desired
    }
}
