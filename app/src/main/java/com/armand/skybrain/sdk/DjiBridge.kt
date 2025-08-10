package com.armand.skybrain.sdk

import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow

class DjiBridge {
    sealed class Status {
        object Disconnected : Status()

        object Connecting : Status()

        data class Connected(val model: String) : Status()
    }

    private val _status = MutableStateFlow<Status>(Status.Disconnected)
    val status: StateFlow<Status> = _status

    fun registerAndConnect(appKey: String) {
        // TODO: DJISDKManager.getInstance().registerApp(...)
        _status.value = Status.Connecting
    }

    fun enableVirtualStick(enable: Boolean) {
        // TODO: flightController?.setVirtualStickModeEnabled(enable, ...)
    }

    fun sendVirtualStick(
        roll: Float,
        pitch: Float,
        yawRate: Float,
        throttle: Float,
    ) {
        // TODO: flightController?.sendVirtualStickFlightControlData(...)
    }
}
