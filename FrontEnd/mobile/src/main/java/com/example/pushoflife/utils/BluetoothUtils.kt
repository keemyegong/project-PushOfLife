package com.example.pushoflife.utils

import android.bluetooth.BluetoothGatt
import android.bluetooth.BluetoothGattCharacteristic
import android.bluetooth.BluetoothGattService
import com.example.pushoflife.bluetooth.CHARACTERISTIC_READ_UUID
import com.example.pushoflife.bluetooth.CHARACTERISTIC_WRITE_UUID
import com.example.pushoflife.bluetooth.SERVICE_UUID
import java.util.*
import kotlin.collections.ArrayList


class BluetoothUtils {
    companion object {
        /**
         * Find characteristics of BLE
         * @param gatt gatt instance
         * @return list of found gatt characteristics
         */
        fun findBLECharacteristics(gatt: BluetoothGatt): List<BluetoothGattCharacteristic> {
            val matchingCharacteristics: MutableList<BluetoothGattCharacteristic> = ArrayList()
            val serviceList = gatt.services
            val service = findGattService(serviceList) ?: return matchingCharacteristics
            val characteristicList = service.characteristics
            for (characteristic in characteristicList) {
                if (isMatchingCharacteristic(characteristic)) {
                    matchingCharacteristics.add(characteristic)
                }
            }
            return matchingCharacteristics
        }

        /**
         * Find command characteristic of the peripheral device
         * @param gatt gatt instance
         * @return found characteristic
         */
        fun findCommandCharacteristic(gatt: BluetoothGatt): BluetoothGattCharacteristic? {
            return findCharacteristic(gatt, CHARACTERISTIC_WRITE_UUID)
        }

        /**
         * Find response characteristic of the peripheral device
         * @param gatt gatt instance
         * @return found characteristic
         */
        fun findResponseCharacteristic(gatt: BluetoothGatt): BluetoothGattCharacteristic? {
            return findCharacteristic(gatt, CHARACTERISTIC_READ_UUID)
        }

        /**
         * Find the given uuid characteristic
         * @param gatt gatt instance
         * @param uuidString uuid to query as string
         */
        private fun findCharacteristic(
            gatt: BluetoothGatt,
            uuidString: String
        ): BluetoothGattCharacteristic? {
            val serviceList = gatt.services
            val service = findGattService(serviceList) ?: return null
            val characteristicList = service.characteristics
            for (characteristic in characteristicList) {
                if (matchCharacteristic(characteristic, uuidString)) {
                    return characteristic
                }
            }
            return null
        }

        /**
         * Match the given characteristic and a uuid string
         * @param characteristic one of found characteristic provided by the server
         * @param uuidString uuid as string to match
         * @return true if matched
         */
        private fun matchCharacteristic(
            characteristic: BluetoothGattCharacteristic?,
            uuidString: String
        ): Boolean {
            if (characteristic == null) {
                return false
            }
            val uuid: UUID = characteristic.uuid
            return matchUUIDs(uuid.toString(), uuidString)
        }

        /**
         * Find Gatt service that matches with the server's service
         * @param serviceList list of services
         * @return matched service if found
         */
        private fun findGattService(serviceList: List<BluetoothGattService>): BluetoothGattService? {
            for (service in serviceList) {
                val serviceUuidString = service.uuid.toString()
                if (matchServiceUUIDString(serviceUuidString)) {
                    return service
                }
            }
            return null
        }

        /**
         * Try to match the given uuid with the service uuid
         * @param serviceUuidString service UUID as string
         * @return true if service uuid is matched
         */
        private fun matchServiceUUIDString(serviceUuidString: String): Boolean {
            return matchUUIDs(serviceUuidString, SERVICE_UUID)
        }

        /**
         * Check if there is any matching characteristic
         * @param characteristic query characteristic
         */
        private fun isMatchingCharacteristic(characteristic: BluetoothGattCharacteristic?): Boolean {
            if (characteristic == null) {
                return false
            }
            val uuid: UUID = characteristic.uuid
            return matchCharacteristicUUID(uuid.toString())
        }

        /**
         * Query the given uuid as string to the provided characteristics by the server
         * @param characteristicUuidString query uuid as string
         * @return true if the matched is found
         */
        private fun matchCharacteristicUUID(characteristicUuidString: String): Boolean {
            return matchUUIDs(
                characteristicUuidString,
                CHARACTERISTIC_WRITE_UUID,
                CHARACTERISTIC_READ_UUID
            )
        }

        /**
         * Try to match a uuid with the given set of uuid
         * @param uuidString uuid to query
         * @param matches a set of uuid
         * @return true if matched
         */
        private fun matchUUIDs(uuidString: String, vararg matches: String): Boolean {
            for (match in matches) {
                if (uuidString.equals(match, ignoreCase = true)) {
                    return true
                }
            }
            return false
        }
    }
}