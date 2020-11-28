import math
import time

class PTZ_Controller_Novel():

    def __init__(self, focal_length = 129, gamma=0.4):
        #Focal Length of a BirdDog P200 is 4.3 to 129mm
        #129mm being the most zoomed out
        #X-axis Speed = 100 deg/s
        #Y-axis Speed = 50 deg/s
        #Pixel size = 89.75 micrometers or 0.08975mm

        self.pix_s = 0.08975
        self.theta_tminus1 = 0
        self.omega_tur_tminus1 = 0
        self.omega_tar_tminus1 = 0
        self.time_prev = None
        self.focal_length = focal_length
        self.gamma = gamma

    def _delta_P(self, obj, center):
        return (obj - center) * self.pix_s

    def _theta(self, delta_P):
        return math.atan((delta_P/self.focal_length))

    def _omega_t(self, theta_present, theta_tminus1, time_prev):
        return theta_present - theta_tminus1 / self._delta_time(self.time_prev)

    def _omega_tplus1(self, theta_t):
        return ((0 - theta_t)/self._delta_time(self.time_prev))*self.gamma

    def _delta_time(self, time_prev, time_now = time.time()):      
        if time_now - time_prev <= 0.03:
            x = 0.03
        else:
            x = time_now - time_prev
        return x

    def omega_tur_plus1(self, obj_coord, center_coord, RMin = 0.0, RMax =10, TMin = 0, TMax =1):
        if self.time_prev == None:
            self.time_prev = time.time()
        else:
            pass
        
        delta_P = self._delta_P(obj_coord, center_coord)
        theta_t = self._theta(delta_P)
        omega_tar_t = self._omega_t(theta_t, self.theta_tminus1, self.time_prev)
        omega_tur_t = self.omega_tur_tminus1
        omega_tar_tplus1 = self._omega_tplus1(theta_t)
        p = omega_tar_t + omega_tur_t + omega_tar_tplus1
        speed = self._angular_to_vector(p, RMin, RMax, TMin, TMax)
        self.present_to_prev(theta_t, omega_tur_t, omega_tar_t)
        #print('Omega Value {}  Speed Value {}'.format(p,speed))
        return speed

    #Translate from angular velocity to camera input vector
    def _angular_to_vector(self, angular_rotation, RMin, RMax, TMin, TMax):
        
        #print('Angular Rotation is: {}'.format(angular_rotation))
        if angular_rotation < 0.2 and angular_rotation > -0.2:
            speed = 0

        else:
            if angular_rotation < 0:
                angular_rotation = abs(angular_rotation)
                speed = (((angular_rotation - RMin) / (RMax - RMin)) * (TMax - TMin)) + TMin
                speed = speed * -1
            else:
                speed = (((angular_rotation - RMin) / (RMax - RMin)) * (TMax - TMin)) + TMin

        return speed

    def present_to_prev(self, theta_t, omega_tur_t, omega_tar_t):
        self.theta_tminus1 = theta_t
        self.omega_tur_tminus1 = omega_tur_t
        self.omega_tar_tminus1 = omega_tar_t
        self.time_prev = time.time()

    def _test(self, m):
        for i in range(0,480):
            val = m.omega_tur_plus1(i,240)
            #time.sleep(0.06)
            #print(val)

def _test():
    m = PTZ_Controller_Novel()
    m._test(m)








