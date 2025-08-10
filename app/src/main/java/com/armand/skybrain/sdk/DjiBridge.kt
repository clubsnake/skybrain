package com.armand.skybrain.sdk

import android.content.Context
import android.util.Log
import dji.common.error.DJIError
import dji.common.flightcontroller.virtualstick.FlightControlData
import dji.common.flightcontroller.virtualstick.RollPitchControlMode
import dji.common.flightcontroller.virtualstick.VerticalControlMode
import dji.common.flightcontroller.virtualstick.YawControlMode
import dji.sdk.base.BaseProduct
import dji.sdk.flightcontroller.FlightController
import dji.sdk.products.Aircraft
import dji.sdk.sdkmanager.DJIBaseComponent
import dji.sdk.sdkmanager.DJISDKManager
import dji.sdk.simulator.InitializationData
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow

/**
 * Bridge between the application and the DJI Mobile SDK.
 *
 * This class encapsulates registration, product connection handling, access to the
 * flight controller and simulator, and exposes a simple status StateFlow for
 * observing connection state. All interactions with DJI classes live in this
 * file to isolate the rest of the app from SDK specifics.
 */
class DjiBridge {
    /** Status of the SDK/product connection. */
    sealed class Status {
        /** No registration attempted or product disconnected. */
        object Disconnected : Status()

        /** Registration in progress. */
        object Connecting : Status()

        /** Product connected with aircraft model name. */
        data class Connected(val model: String) : Status()
    }

    private val _status = MutableStateFlow<Status>(Status.Disconnected)

    /**
     * StateFlow exposing the current connection status. UI code can collect this
     * to enable/disable controls based on whether the aircraft is connected.
     */
    val status: StateFlow<Status> = _status

    private var flightController: FlightController? = null

    /** Shared flight state store.  Collect this from the UI or policy layers
     *  to receive live telemetry updates.  It is started when the product
     *  connects and a flight controller is available. */
    val flightStateStore = FlightStateStore()

    /**
     * Registers the application with the DJI SDK and initiates a connection to the product.
     *
     * @param context Application or Activity context. The App Key is read from
     *        AndroidManifest meta-data (com.dji.sdk.API_KEY). Do not pass null.
     */
    fun registerAndConnect(context: Context) {
        // Avoid duplicate registration attempts.
        if (_status.value != Status.Disconnected) return
        _status.value = Status.Connecting
        DJISDKManager.getInstance()
            .registerApp(
                context.applicationContext,
                object : DJISDKManager.SDKManagerCallback {
                    override fun onRegister(error: DJIError?) {
                        if (error == null) {
                            // Successfully registered; start connection to product.
                            DJISDKManager.getInstance().startConnectionToProduct()
                            Log.i(TAG, "DJI SDK registered. Waiting for product connection...")
                        } else {
                            Log.e(TAG, "DJI SDK registration failed: ${error.description}")
                            _status.value = Status.Disconnected
                        }
                    }

                    override fun onProductDisconnect() {
                        Log.w(TAG, "DJI product disconnected")
                        flightController = null
                        _status.value = Status.Disconnected
                    }

                    override fun onProductConnect(baseProduct: BaseProduct?) {
                        baseProduct?.let { product ->
                            Log.i(TAG, "DJI product connected: ${product.model}")
                            _status.value = Status.Connected(product.model.toString())
                            // Cast to Aircraft to access flight controller
                            val aircraft = product as? Aircraft
                            flightController = aircraft?.flightController
                            flightController?.let { fc ->
                                // Configure virtual stick modes
                                configureVirtualStickModes()
                                // Start listening to telemetry
                                flightStateStore.startListening(fc)
                            }
                        }
                    }

                    override fun onProductChanged(
                        oldProduct: BaseProduct?,
                        newProduct: BaseProduct?,
                    ) {
                        // Treat a product change as a disconnect followed by a connect
                        onProductDisconnect()
                        onProductConnect(newProduct)
                    }

                    // Called when a component's connectivity changes (gimbal, camera, etc.)
                    override fun onComponentChange(
                        componentKey: String?,
                        oldComponent: DJIBaseComponent?,
                        newComponent: DJIBaseComponent?,
                    ) {
                        // No-op for now; can be used to monitor individual component state
                    }
                },
            )
    }

    /**
     * Enables or disables virtual stick mode on the flight controller. If the product
     * is not connected or the flight controller is null, this call is ignored.
     *
     * @param enable true to enable virtual stick, false to disable
     */
    fun enableVirtualStick(enable: Boolean) {
        val fc = flightController ?: return
        fc.setVirtualStickModeEnabled(enable) { error: DJIError? ->
            if (error != null) {
                Log.e(TAG, "Failed to set virtual stick mode: ${error.description}")
            } else {
                Log.i(TAG, "Virtual stick mode set to $enable")
            }
        }
    }

    /**
     * Sends a virtual stick control command. The command uses body‑frame
     * velocities for pitch (forward/back), roll (left/right), yaw rate, and
     * vertical velocity (throttle). If the flight controller is not connected
     * or virtual stick is disabled, the command is ignored.
     *
     * @param roll Left/right velocity in m/s (body frame)
     * @param pitch Forward/back velocity in m/s (body frame)
     * @param yawRate Yaw angular velocity in degrees/s (body frame)
     * @param throttle Vertical velocity in m/s (positive up)
     */
    fun sendVirtualStick(
        roll: Float,
        pitch: Float,
        yawRate: Float,
        throttle: Float,
    ) {
        val fc = flightController ?: return
        val data = FlightControlData(pitch, roll, yawRate, throttle)
        fc.sendVirtualStickFlightControlData(data) { error: DJIError? ->
            if (error != null) {
                Log.e(TAG, "sendVirtualStick failed: ${error.description}")
            }
        }
    }

    /**
     * Starts the DJI Simulator using default initialization parameters. The simulator
     * runs on the aircraft; ensure props are removed and that the aircraft is
     * connected. The simulator does not simulate obstacle sensors.
     */
    fun startSimulator() {
        val fc = flightController ?: return
        val simulator = fc.simulator ?: return
        // Choose an arbitrary starting location (0 lat/lon) and low altitude. Update as needed.
        val initData =
            InitializationData.createInstance(
                SIM_INIT_LAT,
                SIM_INIT_LON,
                SIM_INIT_ALT,
                SIM_INIT_X_VEL,
                SIM_INIT_Y_VEL,
            )
        simulator.start(initData) { error: DJIError? ->
            if (error != null) {
                Log.e(TAG, "Simulator start failed: ${error.description}")
            } else {
                Log.i(TAG, "Simulator started")
            }
        }
    }

    /**
     * Stops the DJI Simulator if it is running. Safe to call if the simulator is
     * not running.
     */
    fun stopSimulator() {
        val fc = flightController ?: return
        val simulator = fc.simulator ?: return
        simulator.stop { error: DJIError? ->
            if (error != null) {
                Log.e(TAG, "Simulator stop failed: ${error.description}")
            } else {
                Log.i(TAG, "Simulator stopped")
            }
        }
    }

    /**
     * Configures the flight controller to use velocity control in the body
     * reference frame for virtual stick. This should be called once upon
     * connection.
     */
    private fun configureVirtualStickModes() {
        val fc = flightController ?: return
        // Roll/pitch as velocity (m/s)
        fc.rollPitchControlMode = RollPitchControlMode.VELOCITY
        // Yaw as angular velocity (deg/s)
        fc.yawControlMode = YawControlMode.ANGULAR_VELOCITY
        // Vertical as velocity (m/s)
        fc.verticalControlMode = VerticalControlMode.VELOCITY
        // Body frame coordinate system
        fc.rollPitchCoordinateSystem = dji.common.flightcontroller.virtualstick.RollPitchCoordinateSystem.BODY
    }

    companion object {
        private const val TAG = "DjiBridge"
        private const val SIM_INIT_LAT = 37.422
        private const val SIM_INIT_LON = -122.084
        private const val SIM_INIT_ALT = 0f
        private const val SIM_INIT_X_VEL = 10f
        private const val SIM_INIT_Y_VEL = 10
    }
}
