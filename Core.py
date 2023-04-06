import os
import sys
from sage.base_app import BaseApp
import time
import numpy as np
from .const import IMU_LIST, IMU_FIELDS, ACC_ALL, GYR_ALL, MAX_BUFFER_LEN, GRAVITY, WEIGHT_LOC, HEIGHT_LOC
third_party_path = os.path.abspath(os.path.join(__file__, '../third_party'))
sys.path.insert(0, third_party_path)

from .GRFPredictor import GRFPredictor

sys.path.remove(third_party_path)

class Core(BaseApp):
    def __init__(self, my_sage):
        BaseApp.__init__(self, my_sage, __file__)
        self.weight = self.config['weight']
        self.height = self.config['height']
        model_path = os.path.join(os.path.dirname(__file__), "models/7IMU_FUSION40_LSTM20.pth")
        scalars_path = os.path.join(os.path.dirname(__file__), "models/scalars.pkl")
        self.grf_predictor = GRFPredictor(model_path, scalars_path, self.weight, self.height)
        self.time_now = 0
    
    def run_in_loop(self):
        data = self.my_sage.get_next_data()
        # Initialize the input dictionaries
        input_acc = [[] for _ in range(len(IMU_LIST))]
        input_gyr = [[] for _ in range(len(IMU_LIST))]

        for d in data:
            sensor = d.get('SensorIndex')
            if sensor in range(len(IMU_LIST)):
                input_acc[sensor].extend([d['AccelX'], d['AccelY'], d['AccelZ']])
                input_gyr[sensor].extend([d['GyroX'], d['GyroY'], d['GyroZ']])
        if input_acc and input_gyr:
            input_acc_dict = {IMU_LIST[sensor]: acc_data for sensor, acc_data in enumerate(input_acc)}
            input_gyr_dict = {IMU_LIST[sensor]: gyr_data for sensor, gyr_data in enumerate(input_gyr)}

            anthro_data = np.zeros([1, 152, 2], dtype=np.float32)
            anthro_data[:, :, WEIGHT_LOC] = self.weight
            anthro_data[:, :, HEIGHT_LOC] = self.height
            x = {'input_acc': input_acc_dict, 'input_gyr': input_gyr_dict, 'anthro': anthro_data}
            data_to_send = self.grf_predictor.update_stream(x)

            for data_item, grf in data_to_send:
                self.time_now += 0.01
                my_data = {'time': [self.time_now], 'plate_1_force_x': [grf], 'plate_1_force_y': [0], 'plate_1_force_z': [0], 'plate_2_force_x': [0], 'plate_2_force_y': [0], 'plate_2_force_z': [0]}
                self.my_sage.send_stream_data(data_item, my_data)
                self.my_sage.save_data(data_item, my_data)

        return True







if __name__ == '__main__':
    # This is only for testing. make sure you do the pairing first in web api
    from sage.sage import Sage
    app = Core(Sage())
    app.test_run()
