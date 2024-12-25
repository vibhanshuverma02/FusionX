import os
import numpy as np
import time
# from sensorfusion import SensorFusion , skew_symmetric
from utils import Quaternion ,SensorFusion , skew_symmetric

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
        
    
    def parse_gps(self, lat=None, lon=None,alt=None, fx=True):
        """
        Get coordinates out of GPS stream
    """
      
        self.sensor_fusion.initialise(np.array([lon,lat,alt]))
        self.imu.initialise()
        self.sim_counter += 1
        
        if (self.imu.initialised):
            self.lon , self.lat, position_corrected, velocity_corrected, quaternion_corrected, p_cov_corrected = self.sensor_fusion.ErrorState_ekf(\
                                                                            position_gnss = np.array([lon,lat,alt]),\
                                                                            position_predict = self.imu.position,\
                                                                            velocity_predict = self.imu.velocity,\
                                                                            quaternion_predict = self.imu.quaternion,\
                                                                            p_cov = self.imu.p_cov)
            self.imu.update(position_corrected, velocity_corrected, quaternion_corrected, p_cov_corrected)
            self.has_fix = fx

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
        
        try:
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

        # Linearize the motion model and compute Jacobians
        self.F = np.eye(9)
        self.F[0,3] = self.F[1,4] = self.F[2,5] = self.delta_t
        self.F[3:6, 6:9] = -skew_symmetric(Quaternion(*self.quaternion).to_mat() @ (self.accel.reshape(3,1))) * self.delta_t

        Q = self.delta_t ** 2 *np.diag([self.var_imu_f,self.var_imu_f,self.var_imu_f,self.var_imu_w,self.var_imu_w,self.var_imu_w])

        # Propagate uncertainty
        self.p_cov = np.dot(np.dot(self.F, self.p_cov), self.F.T) + np.dot(np.dot(self.l_jac, Q), self.l_jac.T)

       
    def parse_imu(self, linear_accel_list = None, quaternion_list = None, angular_vel_list = None):
        """
        ///////////Get heading out of IMU stream
        """
        quaternion_list = np.array(quaternion_list)                         
        q = Quaternion(*quaternion_list).normalize()
        self.quaternion  = q.to_numpy()

        self.accel = np.array(linear_accel_list)
        self.omega = np.array(angular_vel_list)
        if self.initialised:
            # predict the position, velocity and the quaternion for the Sensor Fusion
            self.predict()

        _,_,heading = Quaternion(*quaternion_list).to_euler()
        self.heading = np.rad2deg(-heading)
        self.temperature = 10
        self.has_fix = True
        
    def stop(self):
        self.isStopped = True
        time.sleep(1)
        # self.imu_calibration.close()
        time.sleep(1)
