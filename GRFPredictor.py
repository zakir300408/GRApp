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
        print("Weight: ", self.weight)
        self.height = self.config['height']
        print("Height: ", self.height)
        model_path = os.path.join(os.path.dirname(__file__), "models/7IMU_FUSION40_LSTM20.pth")
        print("Model path: ", model_path)
        scalars_path = os.path.join(os.path.dirname(__file__), "models/scalars.pkl")
        print("Scalars path: ", scalars_path)
        self.grf_predictor = GRFPredictor(model_path, scalars_path, self.weight, self.height)
        print("Predictions made")
        self.time_now = 0

    def run_in_loop(self):
        data = self.my_sage.get_newest_data()
        print("Data: ", data)
        # Extract the input data for each sensor
        input_acc = {sensor: [d['AccelX'], d['AccelY'], d['AccelZ']] for sensor in IMU_LIST for d in data if
                     'SensorIndex' in d and d['SensorIndex'] == sensor}
        print("Input acc: ", input_acc)
        input_gyr = {sensor: [d['GyroX'], d['GyroY'], d['GyroZ']] for sensor in IMU_LIST for d in data if
                     'SensorIndex' in d and d['SensorIndex'] == sensor}
        print("Input gyr: ", input_gyr)

        # Set the anthro data with weight and height
        anthro_data = np.zeros([1, 152, 2], dtype=np.float32)
        anthro_data[:, :, WEIGHT_LOC] = self.weight
        anthro_data[:, :, HEIGHT_LOC] = self.height
        print("Anthro data: ", anthro_data)
        # Create the input dictionary
        x = {'input_acc': input_acc, 'input_gyr': input_gyr, 'anthro': anthro_data}
        data_to_send = self.grf_predictor.update_stream(x)
        print("Data to send: ", data_to_send)

        for data_item, grf in data_to_send:
            self.time_now += 0.01
            my_data = {'time': [self.time_now], 'plate_1_force_x': [grf], 'plate_1_force_y': [0], 'plate_1_force_z': [0], 'plate_2_force_x': [0], 'plate_2_force_y': [0], 'plate_2_force_z': [0]}
            self.my_sage.send_stream_data(data_item, my_data)
            print("Sending data: ", my_data)
            self.my_sage.save_data(data_item, my_data)
            print("Saving data: ", my_data)

        return True

if __name__ == '__main__':
    # This is only for testing. make sure you do the pairing first in web api
    from sage.sage import Sage
    app = Core(Sage())
    app.test_run()
