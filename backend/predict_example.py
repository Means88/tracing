from anomalyDetector import anomalyScore
from run_model import predict as f_pred
import numpy as np


def predict_example(data):
    data = data.transpose()
    interval = 99
    input = np.array(data[-interval:]).reshape(99, 1, -1)
    return f_pred(input, data_dir).reshape(20, -1).transpose()


if __name__ == '__main__':
    data_dir = 'data_np_0.npy'
    data = np.load(data_dir)[:200].transpose()
    print(data.shape)
    print(predict_example(data))

