import numpy as np
import os
import pyproj
import scipy

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
        