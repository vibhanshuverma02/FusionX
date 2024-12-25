import os
import time 
from sensor import IMU, GPS
from can_parser import CANReceiver

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
        self.CANReceiver = CANReceiver(self.config ,self.IMU ,self.GPS)
    
    
    def start(self):
      
        self.CANReceiver.start()
    
    def stop(self):
      
        self.CANReceiver.stop()
 

can_config = {
    "can": {
        "channel": "can1",  # Replace with your channel
        "bus": "socketcan",  # Replace with your bus type
        "bitrate": 500000    # Replace with your desired bitrate
    }
}

if __name__ == "__main__":
   
    sensors = Sensors(can_config)
    

    try:
        sensors.start()
        time_start = time.time()
        while True:
            
            time.sleep(0.01)
            
    except KeyboardInterrupt:
        sensors.stop()
