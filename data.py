import os
import numpy as np
import scipy.io as sio
from paddle.io import Dataset
from sklearn.preprocessing import MinMaxScaler

class Multi_view_data(Dataset):
    """
    load multi-view data
    """

    def __init__(self, root, train=True):
        """
        :param root: data name and path
        :param train: load training set or test set
        """
        super(Multi_view_data, self).__init__()
        self.root = root
        self.train = train
        data_path = self.root + '.mat'

        dataset = sio.loadmat(data_path)
        view_number = int((len(dataset) - 5) / 2)
        self.X = dict()
        if train:
            for v_num in range(view_number):
                self.X[v_num] = normalize(dataset['x' + str(v_num + 1) + '_train'])
            y = dataset['gt_train']
        else:
            for v_num in range(view_number):
                self.X[v_num] = normalize(dataset['x' + str(v_num + 1) + '_test'])
            y = dataset['gt_test']

        if np.min(y) == 1:
            y = y - 1
        tmp = np.zeros(y.shape[0])
        y = np.reshape(y, np.shape(tmp))
        self.y = y

    def __getitem__(self, index):
        data = dict()
        for v_num in range(len(self.X)):
            data[v_num] = (self.X[v_num][index]).astype(np.float32)
        target = self.y[index]
        return data, target

    def __len__(self):
        return len(self.X[0])


def normalize(x, min=0):
    if min == 0:
        scaler = MinMaxScaler([0, 1])
    else:  # min=-1
        scaler = MinMaxScaler((-1, 1))
    norm_x = scaler.fit_transform(x)
    return norm_x


def get_tiny_test_sample(data_path):
    test_set = Multi_view_data(data_path, train=False)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
    res={}
    data_tiny={}
    for data,target in test_loader:
        for k,v in data.items():
            data_tiny[k]=v.numpy()
        res["data"]=data_tiny
        res["target"]=target.numpy()
        break
    np.save("tiny_sample.npy",[res])


def draw_sample0(data_path):
    data_path = data_path+ '.mat'

    dataset = sio.loadmat(data_path)
    view_number = int((len(dataset) - 5) / 2)
    data = dict()
    for v_num in range(view_number):
        data[v_num] = dataset['x' + str(v_num + 1) + '_test']
    y = dataset['gt_test']
    data0={}
    y0=y[0]
    for k,v in data.items():
        data0[k]=v[0]
        # print(data0[k].shape,k)
    import matplotlib.pyplot as plt
    min_val=min(data0[0])
    max_val=max(data0[0])
    data0[0]=255*(data0[0]-min_val)/(max_val-min_val)
    plt.imshow(data0[0].reshape([12,20]))
    plt.show()
    return data,y


if __name__ == '__main__':
    from paddle.io import DataLoader
    data_path="datasets/handwritten_6views"
    # get_tiny_test_sample(data_path=data_path)
    data,y=draw_sample0(data_path)
