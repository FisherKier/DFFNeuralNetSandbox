import logging
import os
import numpy as np
import struct
import matplotlib.pyplot as plt
import cpuinfo

log = logging.getLogger("my-logger")

def parse_mnist_data(data_type, config=-1):
    file_name_dict = {
        "test_images": "t10k-images.idx3-ubyte",
        "test_labels": "t10k-labels.idx1-ubyte",
        "training_images": "train-images.idx3-ubyte",
        "training_labels": "train-labels.idx1-ubyte"
    }

    #check if user inputted data type is in the struct

    file_name = file_name_dict[data_type]
    dir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.path.pardir, "data")
    file_path = os.path.join(dir_path, file_name)

    # if config.endian is null
    cpu_brand = cpuinfo.get_cpu_info()['brand_raw']
    low_endian = False
    if "Intel" in cpu_brand:
        low_endian = True
    #set config.endian, and write to config

    #else low_endian = config.endian ? True : False

    with open(file=file_path, mode='rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        nrows, ncols = struct.unpack(">II", f.read(8))
        data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        data = data.reshape((size, nrows, ncols))
        return data

def test_method():
    data = parse_mnist_data("test_images")

    plt.imshow(data[0,:,:], cmap='gray')
    plt.show()

test_method()