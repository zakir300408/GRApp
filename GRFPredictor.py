import os
from collections import deque
import copy
import random
import numpy as np
import pickle
from .const import IMU_LIST, IMU_FIELDS, ACC_ALL, GYR_ALL, MAX_BUFFER_LEN, GRAVITY, WEIGHT_LOC, HEIGHT_LOC
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch


lstm_unit, fcnn_unit = 100, 200


class InertialNet(nn.Module):
    def __init__(self, x_dim, net_name, seed=0, nlayer=1):
        super(InertialNet, self).__init__()
        self.net_name = net_name
        torch.manual_seed(seed)
        self.rnn_layer = nn.LSTM(x_dim, globals()['lstm_unit'], nlayer, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.4)       # !!!
        for name, param in self.rnn_layer.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def __str__(self):
        return self.net_name

    def forward(self, sequence, lens):
        sequence = pack_padded_sequence(sequence, lens, batch_first=True, enforce_sorted=False)
        sequence, _ = self.rnn_layer(sequence)
        sequence, _ = pad_packed_sequence(sequence, batch_first=True, total_length=152)
        sequence = self.dropout(sequence)
        return sequence


class OutNet(nn.Module):
    def __init__(self, input_dim, device, output_dim=6, high_level_locs=[2, 3, 4]):  # Changed output_dim to 6
        super(OutNet, self).__init__()
        self.high_level_locs = high_level_locs
        self.linear_1 = nn.Linear(input_dim + len(high_level_locs), globals()['fcnn_unit'], bias=True).to(device)
        self.linear_2 = nn.Linear(globals()['fcnn_unit'], output_dim, bias=True).to(device)
        self.relu = nn.ReLU().to(device)
        for layer in [self.linear_1, self.linear_2]:
            nn.init.xavier_normal_(layer.weight)

    def forward(self, sequence, others):
        if len(self.high_level_locs) > 0:
            sequence = torch.cat((sequence, others[:, :, self.high_level_locs]), dim=2)
        sequence = self.linear_1(sequence)
        sequence = self.relu(sequence)
        sequence = self.linear_2(sequence)
        weight = others[:, 0, WEIGHT_LOC].unsqueeze(1).unsqueeze(2)
        height = others[:, 0, HEIGHT_LOC].unsqueeze(1).unsqueeze(2)
        sequence = torch.div(sequence, weight * GRAVITY * height / 100)
        return sequence

import torch.nn.functional as F

class LmfImuOnlyNet(nn.Module):
    def __init__(self, acc_dim, gyr_dim):
        super(LmfImuOnlyNet, self).__init__()
        self.acc_subnet = InertialNet(acc_dim, 'acc net', seed=0)
        self.gyr_subnet = InertialNet(gyr_dim, 'gyr net', seed=0)
        self.rank = 10
        self.fused_dim = 40
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.acc_factor = Parameter(torch.Tensor(self.rank, 1, 2*globals()['lstm_unit'] + 1, self.fused_dim)).to(self.device)
        self.gyr_factor = Parameter(torch.Tensor(self.rank, 1, 2*globals()['lstm_unit'] + 1, self.fused_dim)).to(self.device)
        self.fusion_weights = Parameter(torch.Tensor(1, self.rank)).to(self.device)
        self.fusion_bias = Parameter(torch.Tensor(1, self.fused_dim)).to(self.device)

        # Added layers
        self.fc1 = nn.Linear(self.fused_dim, self.fused_dim // 2).to(self.device)
        self.dropout1 = nn.Dropout(0.3).to(self.device)
        self.fc2 = nn.Linear(self.fused_dim // 2, self.fused_dim // 4).to(self.device)
        self.dropout2 = nn.Dropout(0.3).to(self.device)

        self.out_net = OutNet(self.fused_dim // 4, self.device, output_dim=6)  # Set the output_dim value to 6
        self.fc_out = nn.Linear(self.fused_dim // 4, 6).to(self.device)  # Change the output dimension to 6
        # init factors
        nn.init.xavier_normal_(self.acc_factor, 10)
        nn.init.xavier_normal_(self.gyr_factor, 10)
        nn.init.xavier_normal_(self.fusion_weights)
        self.fusion_bias.data.fill_(0)

    def __str__(self):
        return 'LMF IMU only net'

    def set_scalars(self, scalars):
        self.scalars = scalars

    def set_fields(self, x_fields):
        self.acc_fields = x_fields['input_acc']
        self.gyr_fields = x_fields['input_gyr']

    def forward(self, acc_x, gyr_x, others, lens):
        acc_h = self.acc_subnet(acc_x, lens)
        gyr_h = self.gyr_subnet(gyr_x, lens)
        batch_size = acc_h.data.shape[0]
        data_type = torch.cuda.FloatTensor

        _acc_h = torch.cat((torch.autograd.Variable(torch.ones(batch_size, acc_h.shape[1], 1).type(data_type),
                                                    requires_grad=False), acc_h), dim=2)
        _gyr_h = torch.cat((torch.autograd.Variable(torch.ones(batch_size, gyr_h.shape[1], 1).type(data_type),
                                                    requires_grad=False), gyr_h), dim=2)

        fusion_acc = torch.matmul(_acc_h, self.acc_factor)
        fusion_gyr = torch.matmul(_gyr_h, self.gyr_factor)
        fusion_zy = fusion_acc * fusion_gyr
        # permute to make batch first
        sequence = torch.matmul(self.fusion_weights, fusion_zy.permute(1, 2, 0, 3)).squeeze(dim=2) + self.fusion_bias

        # Added layers
        sequence = F.relu(self.fc1(sequence))
        sequence = self.dropout1(sequence)
        sequence = F.relu(self.fc2(sequence))
        sequence = self.dropout2(sequence)

        sequence = self.out_net(sequence, others)
        return sequence

    def __deepcopy__(self, memo):
        new_instance = self.__class__.__new__(self.__class__)
        new_instance.__dict__.update(self.__dict__)
        new_instance._parameters = {name: param.clone() for name, param in self._parameters.items()}
        new_instance._buffers = {name: buf.clone() for name, buf in self._buffers.items()}
        new_instance._modules = {name: copy.deepcopy(module) for name, module in self._modules.items()}
        return new_instance


class GRFPredictor:
    def __init__(self, model_path, scalars_path, weight, height):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Instantiate the model
        self.model = LmfImuOnlyNet(21, 21)
        self.data_buffer = deque(maxlen=MAX_BUFFER_LEN)
        # Load the state dictionary into the model
        state_dict = torch.load(model_path, map_location=self.device)  # Load the model
        
        self.model.load_state_dict(state_dict)

        self.model.eval()

        # Set fields and scalars
        self.model.set_fields({'input_acc': ACC_ALL, 'input_gyr': GYR_ALL})

        # Load and set the scalars
        with open(scalars_path, 'rb') as f:
            scalars = pickle.load(f)
        self.model.set_scalars(scalars)
        self._data_scalar = scalars

        self.data_array_fields = [axis + '_' + sensor for sensor in IMU_LIST for axis in IMU_FIELDS]
        self.model.acc_col_loc = [self.data_array_fields.index(field) for field in self.model.acc_fields]
        self.model.gyr_col_loc = [self.data_array_fields.index(field) for field in self.model.gyr_fields]

        self.weight = weight
        self.height = height

        anthro_data = np.zeros([1, 152, 2], dtype=np.float32)
        anthro_data[:, :, WEIGHT_LOC] = self.weight
        anthro_data[:, :, HEIGHT_LOC] = self.height
        self.model_inputs = {'others': torch.from_numpy(anthro_data), 'step_length': None,
                             'input_acc': None, 'input_gyr': None}

    def preprocess_data(self, x):
        preprocessed_x = {}
        for k in x.keys():
            if k in ['input_acc', 'input_gyr']:
                preprocessed_x[k] = {}  # Initialize the key with an empty dictionary
                for imu_k in x[k].keys():
                    preprocessed_x[k][imu_k] = self.normalize_array_separately(x[k][imu_k], k, 'transform')
            else:
                preprocessed_x[k] = x[k]
        return preprocessed_x



    def normalize_array_separately(self, data, name, method, scalar_mode='by_each_column'):
        # Normalize the input data
        assert (scalar_mode in ['by_each_column', 'by_all_columns'])
        input_data = data.copy().astype(np.float32)

        if input_data.ndim == 1:
            input_data = input_data[np.newaxis, :]

        original_shape = input_data.shape
        target_shape = [-1, input_data.shape[2]] if scalar_mode == 'by_each_column' and input_data.ndim > 2 else [-1, 1]

        if input_data.ndim > 2:
            input_data[(input_data == 0.).all(axis=2), :] = np.nan
        else:
            input_data[input_data == 0.] = np.nan

        input_data = input_data.reshape(target_shape)
        scaled_data = getattr(self._data_scalar[name], method)(input_data)
        scaled_data = scaled_data.reshape(original_shape)
        scaled_data[np.isnan(scaled_data)] = 0.
        return scaled_data

    def process_data_to_input_format(self):
        # Preprocess the latest data in the buffer
        preprocessed_data = self.preprocess_data(self.data_buffer[-1])
        self.data_buffer[-1] = preprocessed_data

        data_buffer = np.array(self.data_buffer)

        if data_buffer.ndim == 1:
            data_buffer = data_buffer[np.newaxis, :]

        data_dict = {'input_acc': {}, 'input_gyr': {}}

        # Separate the data for each IMU
        for imu_idx, imu_name in enumerate(IMU_LIST):
            imu_data = data_buffer[:, imu_idx * len(IMU_FIELDS):(imu_idx + 1) * len(IMU_FIELDS)]
            if imu_name in ACC_ALL:
                data_dict['input_acc'][imu_name] = imu_data
            if imu_name in GYR_ALL:
                data_dict['input_gyr'][imu_name] = imu_data

        # Add the anthro data
        anthro_data = np.zeros([1, 152, 2], dtype=np.float32)
        anthro_data[:, :, WEIGHT_LOC] = self.weight
        anthro_data[:, :, HEIGHT_LOC] = self.height
        data_dict['anthro'] = anthro_data

        # Add the input_lens data
        data_dict['input_lens'] = np.array([152], dtype=np.int32)

        return data_dict







    def update_stream(self, new_x):
        print(f"new_x: {new_x}")
        self.data_buffer.extend(new_x)

        if len(self.data_buffer) < MAX_BUFFER_LEN:
            return []  # Not enough data in the buffer yet

        # Check if the buffer is close to its maximum capacity
        if len(self.data_buffer) >= MAX_BUFFER_LEN + 1:
            self.data_buffer = self.data_buffer[-MAX_BUFFER_LEN:]  # Keep only the most recent data up to the maximum buffer length

        x = self.process_data_to_input_format()  # Process the data from the buffer
        x = self.preprocess_data(x)

        if not x['input_acc']:
            return []  # If there is no data in 'input_acc', return an empty list

        acc_x = np.concatenate(list(x['input_acc'].values()), axis=-1)
        gyr_x = np.concatenate(list(x['input_gyr'].values()), axis=-1)

        acc_x = torch.tensor(acc_x, dtype=torch.float32).to(self.device)
        gyr_x = torch.tensor(gyr_x, dtype=torch.float32).to(self.device)

        # Normalize acc_x
        acc_mean = self._data_scalar['input_acc']['mean'].to(self.device)
        acc_std = self._data_scalar['input_acc']['std'].to(self.device)
        normalized_acc_x = (acc_x - acc_mean) / acc_std

        # Normalize gyr_x
        gyr_mean = self._data_scalar['input_gyr']['mean'].to(self.device)
        gyr_std = self._data_scalar['input_gyr']['std'].to(self.device)
        normalized_gyr_x = (gyr_x - gyr_mean) / gyr_std

        with torch.no_grad():
            y_pred = self.model(normalized_acc_x, normalized_gyr_x)
        grf_pred = y_pred.cpu().numpy()

        grf_pred = np.squeeze(grf_pred)

        data_to_send = []
        for i in range(len(new_x)):
            data_item = new_x[i][0][0]['Data_Source']
            grf = grf_pred[i][0]
            data_to_send.append((data_item, grf))

        # Clear the data buffer
        self.data_buffer.clear()

        return data_to_send


    def predict(self, x):
        try:
            # Convert input data to tensors
            acc_x = torch.tensor(x['input_acc'], dtype=torch.float32).to(self.device)
            gyr_x = torch.tensor(x['input_gyr'], dtype=torch.float32).to(self.device)
            others = torch.tensor(x['anthro'], dtype=torch.float32).to(self.device)
            lens = torch.tensor(x['input_lens'], dtype=torch.long).to(self.device)

            # Make predictions
            with torch.no_grad():
                predictions = self.model(acc_x, gyr_x, others, lens)

            # Convert predictions back to NumPy arrays
            predictions = predictions.cpu().numpy()
            return predictions
        except Exception as e:
            print(f"Error in predict function: {str(e)}")
            raise e

    def load_scalars(self, scalars_path):
        with open(scalars_path, 'rb') as f:
            self._data_scalar = pickle.load(f)
