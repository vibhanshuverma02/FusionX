import numpy as np
import os
import can
import struct
import threading
from threading import Thread
from sensor import IMU , GPS
class CANReceiver:
    def __init__(self, config, imu, gps):
        self.config = config
        self.channel = self.config["can"]["channel"]
        self.bustype = self.config["can"]["bus"]
        self.bitrate = self.config["can"]["bitrate"]
        self.bus = can.interface.Bus(channel=self.channel, interface=self.bustype, bitrate=self.bitrate)
        self.running = False
        self.imu = imu  # IMU object
        self.gps = gps
        # Mapping of message IDs to handler functions
        self.message_handlers = {
            0x101: self.handle_longitude,
            0x100: self.handle_latitude,
            0x104: self.handle_alt,
            0x105: self.handle_quaternion_w,
            0x106: self.handle_quaternion_x,
            0x107: self.handle_quaternion_y,
            0x108: self.handle_quaternion_z,
            0x109: self.handle_gyro_x,
            0x110: self.handle_gyro_y,
            0x111: self.handle_gyro_z,
            0x112: self.handle_accel_x,
            0x113: self.handle_accel_y,
            0x114: self.handle_accel_z,
        }


        # Shared state
        self.lat =None
        self.lon =None
        self.alt =None
        self.quaternion_values = np.zeros(4)
        self.quat_flag = np.zeros(4)
        self.gnss_flag = np.zeros(3)
        self.accel_flag = np.zeros(3)
        self.accel = np.zeros(3)
        self.gyro = np.zeros(3)
        self.gyro_flag = np.zeros(3)
        

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self.receive_messages)
        self.thread.start()
        print("CAN Receiver started.")

    def stop(self):
        self.running = False
        self.bus.shutdown()
        print("CAN Receiver stopped.")

    def receive_messages(self):
        try:
            while self.running:
                message = self.bus.recv()
                if message is None:
                    continue

                handler = self.message_handlers.get(message.arbitration_id)
                if handler:
                    handler(message)
                else:
                    pass
        except KeyboardInterrupt:
            self.stop()
    def handle_latitude(self, message):
        try:
            # print('--', np.frombuffer(message.data,dtype=np.uint8))
            binary_representation = self.arr_to_unit(message.data)
            self.lat = self.unit_to_float(binary_representation, 1)
            self.gnss_flag[0]=1
            self.process_gps_data()
        except KeyboardInterrupt:
            self.stop()
    def handle_longitude(self, message):
        try:
            binary_representation = self.arr_to_unit(message.data)
            # print('--- ', binary_representation)
            self.lon = self.unit_to_float(binary_representation, 2)
            self.gnss_flag[1]=1
            self.process_gps_data()
        except KeyboardInterrupt:
            self.stop() 
    def handle_alt(self, message):
        try:
            binary_representation = self.arr_to_unit(message.data)
            self.alt = self.unit_to_float(binary_representation, 5)
            self.gnss_flag[2]=1
            self.process_gps_data()
        except KeyboardInterrupt:
            self.stop()                

            
    def handle_quaternion_w(self, message):
        try:
            binary_representation = self.arr_to_unit(message.data)
            self.quaternion_values[0] = self.unit_to_float(binary_representation, 4)
            self.quat_flag[0] = 1
        except KeyboardInterrupt:
            self.stop()    

    def handle_quaternion_x(self, message):
        try:
            binary_representation = self.arr_to_unit(message.data)
            self.quaternion_values[1] = self.unit_to_float(binary_representation, 4)
            self.quat_flag[1] = 1
        except KeyboardInterrupt:
            self.stop()

    def handle_quaternion_y(self, message):
        try:
            binary_representation = self.arr_to_unit(message.data)
            self.quaternion_values[2] = self.unit_to_float(binary_representation, 4)
            self.quat_flag[2] = 1
        except KeyboardInterrupt:
            self.stop()

    def handle_quaternion_z(self, message):
        try:
            binary_representation = self.arr_to_unit(message.data)
            self.quaternion_values[3] = self.unit_to_float(binary_representation, 4)
            self.quat_flag[3] = 1
        except KeyboardInterrupt:
            self.stop()

    def handle_gyro_x(self, message):
        try:
            binary_representation = self.arr_to_unit(message.data)
            self.gyro[0] = self.unit_to_float(binary_representation, 4) 
            self.gyro_flag[0] = 1
            self.process_imu_data()
        except KeyboardInterrupt:
            self.stop()

    def handle_gyro_y(self, message):
        try:
            binary_representation = self.arr_to_unit(message.data)
            self.gyro[1] = self.unit_to_float(binary_representation, 4) 
            self.gyro_flag[1] = 1
            self.process_imu_data()
        except KeyboardInterrupt:
            self.stop()     

    def handle_gyro_z(self, message):
        try:
            binary_representation = self.arr_to_unit(message.data)
            self.gyro[2] = self.unit_to_float(binary_representation, 4) 
            self.gyro_flag[2] = 1
            self.process_imu_data()
        except KeyboardInterrupt:
            self.stop()

    def handle_accel_x(self, message):
        try:
            binary_representation = self.arr_to_unit(message.data)
            self.accel[0] = self.unit_to_float(binary_representation, 4) 
            self.accel[0] = self.accel_avg(self.accel[0], -0.07)
            self.accel_flag[0] = 1
            self.process_imu_data()
        except KeyboardInterrupt:
            self.stop()
            
    def handle_accel_y(self, message):
        try:
            binary_representation = self.arr_to_unit(message.data)
            self.accel[1] = self.unit_to_float(binary_representation, 4) 
            self.accel[1] = 0.0#self.accel_avg(self.accel[1] , 0.06)
            self.accel_flag[1] = 1
            self.process_imu_data()
        except KeyboardInterrupt:
            self.stop()

    def handle_accel_z(self, message):
        try:
            binary_representation = self.arr_to_unit(message.data)
            # self.accel[2] = self.unit_to_float(binary_representation, 4) 
            self.accel[2] = 0.0#self.accel_avg(self.accel[2] , -0.22)
            self.accel_flag[2] = 1
            self.process_imu_data()
        except KeyboardInterrupt:
            self.stop()      

    def process_imu_data(self):
        try:
            if np.all(self.quat_flag) and  np.all(self.accel_flag) and np.all(self.gyro) is not None:
                self.imu.parse_imu(
                    linear_accel_list=np.array(self.accel),
                    quaternion_list=np.array(self.quaternion_values),
                    angular_vel_list=np.array(self.gyro) ,
                )
                self.accel = np.zeros(3)
                self.gyro = np.zeros(3)
                self.quaternion_values = np.zeros(4)
                self.quat_flag = np.zeros(4)
                self.accel_flag = np.zeros(3)
                self.gyro_flag = np.zeros(3)
        except KeyboardInterrupt:
            self.stop()
    def process_gps_data(self):
        try:
            if np.all(self.gnss_flag == 1):
                print('--',self.lon) 
                self.gps.parse_gps(
                            lat = self.lat, 
                            lon = self.lon,
                            alt = self.alt)
                self.lat = None
                self.lon = None
                self.alt = None
                self.gnss_flag= np.zeros(3)
        except KeyboardInterrupt:
            # print('1111')
            self.stop()
              

    @staticmethod
    def arr_to_unit(arr):
        combined_int = 0
        for i, byte in enumerate(arr):
            combined_int |= byte << (8 * i)
        return combined_int

    @staticmethod
    def unit_to_float(uint_val, num):
        if num == 1:
            return (uint_val / 1e9) - 90.0
        elif num == 2:
            return (uint_val / 1e9) - 180.0
        elif num == 3:
            return (uint_val / 1e2) - 10.0
        elif num == 4:
            return (uint_val / 1e6) - 10.0
        elif num == 5:
            return (uint_val / 1e3)

    @staticmethod
    def decode_two_byte_values(byte_array, length=8):
        decoded_values = []
        for i in range(0, len(byte_array), 2):
            if i == length:
                break
            raw_value = (byte_array[i] << 8) | byte_array[i + 1]
            decoded_values.append(CANReceiver.unit_to_float(raw_value, 3))
            
        return decoded_values

    @staticmethod
    def decode_two_byte_values_accel(byte_array, length=8, seq=2):
       
        decoded_values = 0
        raw_value=0
        
        for i, byte in enumerate(byte_array):
            raw_value |= byte << (8 * i)
            decoded_values=CANReceiver.unit_to_float(raw_value, 3)
        return decoded_values

    def  accel_avg(self, inputvalue ,avg):
        if(inputvalue>=avg):
            return 0
        else:
            return np.sign(inputvalue) * -1 * abs(avg) + inputvalue 
