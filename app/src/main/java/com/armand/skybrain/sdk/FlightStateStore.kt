package com.armand.skybrain.sdk

import dji.common.flightcontroller.FlightControllerState
import dji.common.flightcontroller.ObstacleAvoidanceSensorState
import dji.sdk.flightcontroller.FlightController
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow

private const val MAX_LIGHT_LEVEL = 255f
private const val MIN_OA_DIST_M = 0f

data class FlightState(
    val tMonoNs: Long = 0L,
    val lat: Double = .0,
    val lon: Double = .0,
    val altM: Float = 0f,
    val vx: Float = 0f,
    val vy: Float = 0f,
    val vz: Float = 0f,
    val yawDeg: Float = 0f,
    val pitchDeg: Float = 0f,
    val rollDeg: Float = 0f,
    val gpsFix: Int = 0,
    val vpsValid: Boolean = false,
    val oaValid: Boolean = false,
    val oaDistM: Float = Float.POSITIVE_INFINITY,
    val lightProxy: Float = 0f,
)

class FlightStateStore {
    private val _state = MutableStateFlow(FlightState())
    val state: StateFlow<FlightState> = _state

    fun update(newState: FlightState) {
        _state.value = newState
    }

    fun startListening(fc: FlightController) {
        fc.setStateCallback { s: FlightControllerState ->
            val pos = s.aircraftLocation
            val lat = pos?.latitude ?: 0.0
            val lon = pos?.longitude ?: 0.0
            val alt = pos?.altitude ?: 0f
            val velX = s.velocityX ?: 0f
            val velY = s.velocityY ?: 0f
            val velZ = s.velocityZ ?: 0f
            val vx = -velX
            val vy = velY
            val vz = -velZ
            val yaw = s.attitude.yaw
            val pitch = s.attitude.pitch
            val roll = s.attitude.roll
            val gpsFix = s.gpsSignalLevel?.value ?: 0
            val vpsValid = s.visionPositioningEnabled
            var oaDist = Float.POSITIVE_INFINITY
            var oaValid = false
            val oaSensorState: ObstacleAvoidanceSensorState? = s.obstacleSensorState
            if (
                oaSensorState != null &&
                oaSensorState.isSensorPositionValid(
                    dji.common.flightcontroller.ObstaclePosition.FRONT,
                )
            ) {
                val dist =
                    oaSensorState.getDistance(
                        dji.common.flightcontroller.ObstaclePosition.FRONT,
                    )
                if (dist != null && dist.isFinite() && dist > MIN_OA_DIST_M) {
                    oaDist = dist
                    oaValid = true
                }
            }
            val lightProxy = s.lightLevel?.let { level -> level.toFloat() / MAX_LIGHT_LEVEL } ?: 0f
            val newState =
                FlightState(
                    tMonoNs = System.nanoTime(),
                    lat = lat,
                    lon = lon,
                    altM = alt,
                    vx = vx,
                    vy = vy,
                    vz = vz,
                    yawDeg = yaw,
                    pitchDeg = pitch,
                    rollDeg = roll,
                    gpsFix = gpsFix,
                    vpsValid = vpsValid,
                    oaValid = oaValid,
                    oaDistM = oaDist,
                    lightProxy = lightProxy,
                )
            _state.value = newState
        }
    }
}
