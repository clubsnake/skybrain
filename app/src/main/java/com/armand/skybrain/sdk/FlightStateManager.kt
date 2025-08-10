package com.armand.skybrain.sdk

import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow

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
}
