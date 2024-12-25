import scipy
import struct
import threading
import yaml
import can
import pyproj
import time
# import datetime as dts
from collections import deque
from threading import Thread
import matplotlib.pyplot as plt
import numpy as np
import os

x_values = deque()
y_values = deque()

"""
Average X: -0.07
Average Y: 0.04
Average Z: -0.22
"""


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
        # if self.thread:
        #     self.thread.join()
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
                    # print(f"Unhandled message ID: {message.arbitration_id}")
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
        

        # for i in range(2):
        #     # print('\nAAAAAAAAAA' , self.decode_two_byte_values_accel(message.data[i*3:(i+1)*3], 3))
        #     self.accel[i] = self.decode_two_byte_values_accel(message.data[i*3:(i+1)*3], 3)
        #     self.accel_flag[i] = 1
        #     self.process_imu_data()
        # # except KeyboardInterrupt:
        # #     self.stop()    
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
                # print('--- ', self.gyro)
                self.imu.parse_imu(
                    linear_accel_list=np.array(self.accel),
                    quaternion_list=np.array(self.quaternion_values),
                    angular_vel_list=np.array(self.gyro) ,
                )
                # self.filename = 'imu_log.txt'
                # current_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
                # with open(self.filename,'a+') as file:
                #     file.write(f'{current_time}  {self.accel[0]} {self.accel[1]} {self.accel[2]} \n')
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
            # print('gyro', decoded_values)
        return decoded_values

    @staticmethod
    def decode_two_byte_values_accel(byte_array, length=8, seq=2):
        # print(len(byte_array))
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
class GPS:
    """
    Class to handle GPS data, it could be from simulation or from the Sensor
    """
    def __init__(self, imu):
        self.sim_counter = 0
        self.imu = imu
        self.sensor_fusion = SensorFusion()
    
        # NOTE: Configure the GPS data formats and frequency here
                
        self.lat = None
        self.lon = None
        self.alt = None
        self.h_accu = 12    # dummy number
        self.has_fix = True
        self.isStopped = False
        self.heading = 0
        self.run = 0
        self.counter = 1
        # self.success = True
        files = os.listdir('/home/debian/Sensor_test/GPS')
        matching_files = [file for file in files if file.startswith('GPSrun_') and file.endswith('.txt')]
        for file in matching_files:
            r = file.split('_')[-1].split('.txt')[0]
            if r.isdigit():
                part = int(r)
                if self.run < part:
                    self.run = part
        self.filename = f'/home/debian/Sensor_test/GPS/GPSrun_{self.run + 1}.txt'
        
    
    # def start(self):
    #     Thread(target=self.parse_gps, args=()).start()
    #     return self
    
    def parse_gps(self, lat=None, lon=None,alt=None, fx=True):
        """
        Get coordinates out of GPS stream
    """
        # print("lat",lat)
    # try :    
        global x_values, y_values
        if self.sim_counter == 0:
            # print('\n\n\nlon', [lon,lat,alt], '\n\n\n')
            self.sensor_fusion.initialise(np.array([lon,lat,alt]))
            self.imu.initialise()
            self.sim_counter += 1
        # print("-1-" , self.imu.quaterion)
        if (self.imu.initialised):
            # print("11",self.imu.quaternion)
            # Inject the GPS values to get a sensor fused corrected position, velocity etc.
            self.lon , self.lat, position_corrected, velocity_corrected, quaternion_corrected, p_cov_corrected = self.sensor_fusion.ErrorState_ekf(\
                                                                            position_gnss = np.array([lon,lat,alt]),\
                                                                            position_predict = self.imu.position,\
                                                                            velocity_predict = self.imu.velocity,\
                                                                            quaternion_predict = self.imu.quaternion,\
                                                                            p_cov = self.imu.p_cov)
            # print ('\n latitude',lat)
            # print('\n longitude' ,lon )                                                                
            # write to a file for making comparisions
            '''
            Format: raw_lat  raw_lon  filtered_lat  filtered_lon
            '''
            with open(self.filename,'a+') as file:
                file.write(f'{lat}  {lon}  {self.lat}  {self.lon}  \n')

            # if self.counter % 3 == 0:
            x_values.append(self.lat)
            y_values.append(self.lon)
            #print('11',x_values)
            # plot_values()
            self.counter += 1    
            
            # Use the corrected position obtained from the sensor fusion to further keep updating the imu prediction
            self.imu.update(position_corrected, velocity_corrected, quaternion_corrected, p_cov_corrected)
            # print('11',position_corrected)
            # self.lat, self.lon = lat, lon
            self.has_fix = fx
        # except Exception as e:
        #     print(f"!!!!  {e}")
        #     pass    
            
        
    def stop(self):
        self.isStopped = True



class IMU:
    """
    Class to handle IMU data, it could be from simulation or from the Sensor
    """
    def __init__(self):
        self.sim_counter = 0
        self.time_now = time.time()
        self.counter = 0
        self.delta_t = None
        self.var_imu_f = 0.10
        self.var_imu_w = 0.25
        self.l_jac = np.zeros([9, 6])
        self.l_jac[3:, :] = np.eye(6)  # motion model noise jacobian

        self.g = np.array([0, 0, -9.81])  # gravity
        self.accel_xyz = np.empty((0, 3))
        self.omega_xyz = np.empty((0, 3))

        self.initialised = False
        self.run = 0
        files = os.listdir('/home/debian/Sensor_test/plot')
        matching_files = [file for file in files if file.startswith('IMUrun_') and file.endswith('.txt')]
        for file in matching_files:
            r = file.split('_')[-1].split('.txt')[0]
            if r.isdigit():
                part = int(r)
                if self.run < part:
                    self.run = part
        self.filename = f'/home/debian/Sensor_test/plot/IMUrun_{self.run + 1}.txt'
        
        try:
            # self.imu = _IMU.Load(name="Load", portID=8)     
            
            # set calibration parameters
            # NOTE: write your self calibration logic here but be careful to name the imu object as self.imu
            
            self.success = True
            
        except:
            self.success = False
        
            
        self.quaternion = np.zeros(4)
        self.accel = np.array([0.0,0.0,0.0])
        self.omega = np.array([0.0,0.0,0.0])
      
        self.accel_enu = None
        self.heading = None
        self.has_fix = False
        self.isStopped = False
        self.temperature = None
        self.quaternion_list = None
        

        self.counter = 1        
    
    def initialise(self):
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.p_cov = np.zeros([9, 9])  # covariance of estimate
        self.initialised = True
        self.quaternion = np.zeros(4)

    def update(self,position,velocity,quaternion,p_cov):
        self.position = position
        self.velocity = velocity
        self.quaternion = quaternion
        self.p_cov = p_cov  # covariance of estimate

    def predict(self):
        # global x_values, y_values
        self.delta_t = time.time() - self.time_now
        self.time_now = time.time()
        omega_t = self.omega * self.delta_t
        # print(f"Omega: {omega_t}")
        delta_q = Quaternion(axis_angle = omega_t).to_numpy()
        q_est_temp = Quaternion(*self.quaternion)

        # Update state with IMU inputs
        self.position = self.position + self.velocity * self.delta_t + 0.5 * self.delta_t ** 2 * (q_est_temp.to_mat() @ self.accel + self.g)
        self.velocity = self.velocity + self.delta_t * (q_est_temp.to_mat() @ self.accel + self.g)
        self.quaternion = q_est_temp.quat_mult_left(delta_q)

        # print(f"---------- {self.position}  {self.accel}")

        # Linearize the motion model and compute Jacobians
        self.F = np.eye(9)
        self.F[0,3] = self.F[1,4] = self.F[2,5] = self.delta_t
        self.F[3:6, 6:9] = -skew_symmetric(Quaternion(*self.quaternion).to_mat() @ (self.accel.reshape(3,1))) * self.delta_t

        Q = self.delta_t ** 2 *np.diag([self.var_imu_f,self.var_imu_f,self.var_imu_f,self.var_imu_w,self.var_imu_w,self.var_imu_w])

        # Propagate uncertainty
        self.p_cov = np.dot(np.dot(self.F, self.p_cov), self.F.T) + np.dot(np.dot(self.l_jac, Q), self.l_jac.T)

        # with open(self.filename,'a+') as file:
        #         file.write(f'{self.position[0]} {self.position[1]} {self.position[2]} \n')
            
        # if self.counter % 3 == 0:
        #     x_values.append(self.position[0])
        #     y_values.append(self.position[1])
        #     # self.plot_values()
        # self.counter += 1
        
            
    def start(self):  
        Thread(target=self.parse_imu, args=()).start()
        return self
 
    def parse_imu(self, linear_accel_list = None, quaternion_list = None, angular_vel_list = None):
        """
        ///////////Get heading out of IMU stream
        """

        # print(f"Quaternion: {quaternion_list}")
        quaternion_list = np.array(quaternion_list)                         
        q = Quaternion(*quaternion_list).normalize()
        self.quaternion  = q.to_numpy()

        self.accel = np.array(linear_accel_list)
        # print(f'accel :{angular_vel_list}')     #these are the acceleration in ENU coordinates
        self.omega = np.array(angular_vel_list)
        # print( 'angular_velocity', self.omega)
        # if self.counter % 3 == 0:
        #     x_values.append(self.counter)
        #     y_values.append(self.omega[2])
        # self.counter += 1
        if self.initialised:
            # predict the position, velocity and the quaternion for the Sensor Fusion
            self.predict()

        _,_,heading = Quaternion(*quaternion_list).to_euler()
        self.heading = np.rad2deg(-heading)
        # print(f'Heading from the quaternion: {heading} -> {self.heading}\n\n')
        self.temperature = 10
        self.has_fix = True

    # def save(self):
    #     self.imu_calibration.close()
        
    def stop(self):
        self.isStopped = True
        time.sleep(1)
        # self.imu_calibration.close()
        time.sleep(1)


class Transforms:

    def __init__(self,lon_org, lat_org, alt_org):
        self.transformer = pyproj.Transformer.from_crs(
            {"proj":'latlong', "ellps":'WGS84', "datum":'WGS84'},
            {"proj":'geocent', "ellps":'WGS84', "datum":'WGS84'},
            )
        self.transformer2 = pyproj.Transformer.from_crs(
            {"proj":'geocent', "ellps":'WGS84', "datum":'WGS84'},
            {"proj":'latlong', "ellps":'WGS84', "datum":'WGS84'},
            )
        self.x_org, self.y_org, self.z_org = self.transformer.transform( lon_org,lat_org,  alt_org,radians=False)
        rot1 =  scipy.spatial.transform.Rotation.from_euler('x', -(90-lat_org), degrees=True).as_matrix()#angle*-1 : left handed *-1
        rot3 =  scipy.spatial.transform.Rotation.from_euler('z', -(90+lon_org), degrees=True).as_matrix()#angle*-1 : left handed *-1

        self.rotMatrix = rot1.dot(rot3)
        self.ecef_org = np.array([[self.x_org,self.y_org,self.z_org]]).T

    def geodetic2enu(self,lon, lat, alt):

        x, y, z = self.transformer.transform( lon,lat,  alt,radians=False)

        vec=np.array([[ x-self.x_org, y-self.y_org, z-self.z_org]]).T

        enu = self.rotMatrix.dot(vec).T.ravel()
        return enu.T

    def enu2geodetic(self,x,y,z):

        ecefDelta = self.rotMatrix.T.dot( np.array([[x,y,z]]).T )
        ecef = ecefDelta + self.ecef_org
        lon, lat, alt = self.transformer2.transform( ecef[0,0],ecef[1,0],ecef[2,0],radians=False)

        return [lon,lat,alt]

class SensorFusion:

    def __init__(self):
        # Initialise the parameters
        self.lat_filtered = None
        self.lon_filtered = None

        self.var_gnss  = 0.01
        self.h_jac = np.zeros([3, 9])
        self.h_jac[:, :3] = np.eye(3)  # measurement model jacobian

    def initialise(self,position):
        # Initializes the transformation ENU<->LLA (Latitude,Longitude,Altitude)
        # print('\n position', position)
        self.geo_transform = Transforms(*position)


    def measurement_update(self,sensor_var, p_cov_check, y_k, p_check, v_check, q_check):
        # Compute Kalman Gain
        V = sensor_var * np.eye(3)
        K = np.dot(np.dot(p_cov_check, self.h_jac.T), np.linalg.inv(np.dot(np.dot(self.h_jac, p_cov_check), self.h_jac.T) + V))

        # Compute error state
        delta_x = K @ (y_k - p_check)

        # Correct predicted state
        p_hat = p_check + delta_x[0:3]
        v_hat = v_check + delta_x[3:6]
        delta_phi = delta_x[6:]
        # print("q",delta_phi, q_check)
        q_hat = Quaternion(euler=delta_phi).quat_mult_left(q_check)

        # Compute corrected covariance
        p_cov_hat = np.dot((np.eye(p_cov_check.shape[0]) - np.dot(K,self.h_jac)),p_cov_check)

        return p_hat, v_hat, q_hat, p_cov_hat

    def ErrorState_ekf(self,position_gnss, position_predict,velocity_predict,quaternion_predict, p_cov):
        # Convert the unfiltered GPS sensor coordinates to ENU
        position_gnss = self.geo_transform.geodetic2enu(*position_gnss)

        position_predict, velocity_predict, quaternion_predict, p_cov = \
            self.measurement_update(self.var_gnss, p_cov, position_gnss.T, position_predict, velocity_predict, quaternion_predict)

        # Convert the filtered ENU coordinates to LLA format
        self.lon_filtered, self.lat_filtered,_ = self.geo_transform.enu2geodetic(*position_predict)

        return self.lon_filtered, self.lat_filtered, position_predict, velocity_predict, quaternion_predict, p_cov


def angle_normalize(a):
    """Normalize angles to lie in range -pi < a[i] <= pi."""
    a = np.remainder(a, 2*np.pi)
    a[a <= -np.pi] += 2*np.pi
    a[a  >  np.pi] -= 2*np.pi
    return a

def skew_symmetric(v):
    """Skew symmetric form of a 3x1 vector."""
    return np.array(
        [[0, -v[2][0], v[1][0]],
         [v[2][0], 0, -v[0][0]],
         [-v[1][0], v[0][0], 0]], dtype=np.float64)

def rpy_jacobian_axis_angle(a):
    """Jacobian of RPY Euler angles with respect to axis-angle vector."""
    if not (type(a) == np.ndarray and len(a) == 3):
        raise ValueError("'a' must be a np.ndarray with length 3.")
    # From three-parameter representation, compute u and theta.
    na  = np.sqrt(a @ a)
    na3 = na**3
    t = np.sqrt(a @ a)
    u = a/t

    # First-order approximation of Jacobian wrt u, t.
    Jr = np.array([[t/(t**2*u[0]**2 + 1), 0, 0, u[0]/(t**2*u[0]**2 + 1)],
                   [0, t/np.sqrt(1 - t**2*u[1]**2), 0, u[1]/np.sqrt(1 - t**2*u[1]**2)],
                   [0, 0, t/(t**2*u[2]**2 + 1), u[2]/(t**2*u[2]**2 + 1)]])

    # Jacobian of u, t wrt a.
    Ja = np.array([[(a[1]**2 + a[2]**2)/na3,        -(a[0]*a[1])/na3,        -(a[0]*a[2])/na3],
                   [       -(a[0]*a[1])/na3, (a[0]**2 + a[2]**2)/na3,        -(a[1]*a[2])/na3],
                   [       -(a[0]*a[2])/na3,        -(a[1]*a[2])/na3, (a[0]**2 + a[1]**2)/na3],
                   [                a[0]/na,                 a[1]/na,                 a[2]/na]])

    return Jr @ Ja

class Quaternion():
    def __init__(self, w=1., x=0., y=0., z=0., axis_angle=None, euler=None):
        """
        Allow initialization with explicit quaterion wxyz, axis-angle, or Euler XYZ (RPY) angles.
        """
        if axis_angle is None and euler is None:
            self.w = w
            self.x = x
            self.y = y
            self.z = z
        elif euler is not None and axis_angle is not None:
            raise AttributeError("Only one of axis_angle or euler can be specified.")
        elif axis_angle is not None:
            if not (type(axis_angle) == list or type(axis_angle) == np.ndarray) or len(axis_angle) != 3:
                raise ValueError("axis_angle must be list or np.ndarray with length 3.")
            axis_angle = np.array(axis_angle)
            norm = np.linalg.norm(axis_angle)
            self.w = np.cos(norm / 2)
            if norm < 1e-50:  # to avoid instabilities and nans
                self.x = 0
                self.y = 0
                self.z = 0
            else:
                imag = axis_angle / norm * np.sin(norm / 2)
                self.x = imag[0].item()
                self.y = imag[1].item()
                self.z = imag[2].item()
        else:
            roll = euler[0]
            pitch = euler[1]
            yaw = euler[2]

            cy = np.cos(yaw * 0.5)
            sy = np.sin(yaw * 0.5)
            cr = np.cos(roll * 0.5)
            sr = np.sin(roll * 0.5)
            cp = np.cos(pitch * 0.5)
            sp = np.sin(pitch * 0.5)

            # Fixed frame
            self.w = cr * cp * cy + sr * sp * sy
            self.x = sr * cp * cy - cr * sp * sy
            self.y = cr * sp * cy + sr * cp * sy
            self.z = cr * cp * sy - sr * sp * cy

            # Rotating frame
            # self.w = cr * cp * cy - sr * sp * sy
            # self.x = cr * sp * sy + sr * cp * cy
            # self.y = cr * sp * cy - sr * cp * sy
            # self.z = cr * cp * sy + sr * sp * cy

    def __repr__(self):
        return "Quaternion (wxyz): [%2.5f, %2.5f, %2.5f, %2.5f]" % (self.w, self.x, self.y, self.z)

    def to_axis_angle(self):
        t = 2*np.arccos(self.w)
        return np.array(t*np.array([self.x, self.y, self.z])/np.sin(t/2))

    def to_mat(self):
        v = np.array([self.x, self.y, self.z]).reshape(3,1)
        return (self.w ** 2 - np.dot(v.T, v)) * np.eye(3) + \
               2 * np.dot(v, v.T) + 2 * self.w * skew_symmetric(v)

    def to_euler(self):
        """Return as xyz (roll pitch yaw) Euler angles."""
        roll = np.arctan2(2 * (self.w * self.x + self.y * self.z), 1 - 2 * (self.x**2 + self.y**2))
        pitch = np.arcsin(2 * (self.w * self.y - self.z * self.x))
        yaw = np.arctan2(2 * (self.w * self.z + self.x * self.y), 1 - 2 * (self.y**2 + self.z**2))
        return np.array([roll, pitch, yaw])

    def to_numpy(self):
        """Return numpy wxyz representation."""
        return np.array([self.w, self.x, self.y, self.z])

    def normalize(self):
        """Return a (unit) normalized version of this quaternion."""
        norm = np.linalg.norm([self.w, self.x, self.y, self.z])
        return Quaternion(self.w / norm, self.x / norm, self.y / norm, self.z / norm)

    def quat_mult_right(self, q, out='np'):
        """
        Quaternion multiplication operation - in this case, perform multiplication
        on the right, that is, q*self.
        """
        v = np.array([self.x, self.y, self.z]).reshape(3, 1)
        sum_term = np.zeros([4,4])
        sum_term[0,1:] = -v[:,0]
        sum_term[1:, 0] = v[:,0]
        sum_term[1:, 1:] = -skew_symmetric(v)
        sigma = self.w * np.eye(4) + sum_term

        if type(q).__name__ == "Quaternion":
            quat_np = np.dot(sigma, q.to_numpy())
        else:
            quat_np = np.dot(sigma, q)

        if out == 'np':
            return quat_np
        elif out == 'Quaternion':
            quat_obj = Quaternion(quat_np[0], quat_np[1], quat_np[2], quat_np[3])
            return quat_obj

    def quat_mult_left(self, q, out='np'):
        """
        Quaternion multiplication operation - in this case, perform multiplication
        on the left, that is, self*q.
        """
        v = np.array([self.x, self.y, self.z]).reshape(3, 1)
        sum_term = np.zeros([4,4])
        sum_term[0,1:] = -v[:,0]
        sum_term[1:, 0] = v[:,0]
        sum_term[1:, 1:] = skew_symmetric(v)
        sigma = self.w * np.eye(4) + sum_term

        if type(q).__name__ == "Quaternion":
            quat_np = np.dot(sigma, q.to_numpy())
        else:
            quat_np = np.dot(sigma, q)

        if out == 'np':
            return quat_np
        elif out == 'Quaternion':
            quat_obj = Quaternion(quat_np[0], quat_np[1], quat_np[2], quat_np[3])
            return quat_obj
        

can_config = {
    "can": {
        "channel": "can1",  # Replace with your channel
        "bus": "socketcan",  # Replace with your bus type
        "bitrate": 500000    # Replace with your desired bitrate
    }
}

class Sensors:
    """
    Class to handle sensors GPS and IMU
    """
    def __init__(self , can_config):
        """
        Init GPS and IMU Sensors
        """
        self.IMU = IMU()
        self.IMU.initialise()
        self.GPS = GPS(self.IMU)
        self.config = can_config
        self.CANReceiver = CANReceiver(self.config, self.IMU ,self.GPS)
    
        # if self.GPS.success and self.IMU.success:
        #     self.success = True
        # else:
        #     self.success = False  
    
    def start(self):
        # start IMU thread
        # self.IMU.start()
        # self.GPS.start()
        self.CANReceiver.start()
    
    def stop(self):
        # self.GPS.stop()
        # start IMU thread
        # self.IMU.stop()
        # Stop CANReceiver thread
        self.CANReceiver.stop()

plt.ion()
fig,ax = plt.subplots()
line, = ax.plot([], [], 'b-')

# Function to plot the data in real-time
def plot_values(max_points=1000, update_interval=0.05):
    """Plot data in real-time with decimation."""
    

    # while True:
    #     with lock:
            # Decimate data for plotting
    # x_dec, y_dec = list(x_values), list(y_values), max_points=max_points)
    # print(len(x_values))
    # try :
    if len(x_values) ==len(y_values):
        line.set_xdata(x_values)
        line.set_ydata(y_values )
        ax.relim()
        ax.autoscale_view()

        fig.canvas.draw()
        fig.canvas.flush_events()
    else:
        print('chutiya')
    # except:
    #     pass        
    # time.sleep(update_interval)  

if __name__ == "__main__":
    # with open('parameter.yaml', 'r') as file:
    #     param = yaml.safe_load(file)

    sensors = Sensors(can_config)
    

    try:
        sensors.start()
        time_start = time.time()
        while True:
            plot_values()
            # print(x_values)
            time.sleep(0.01)
            
    except KeyboardInterrupt:
        sensors.stop()
