from typing import List, Optional
from ..data_basic import Dataset
import numpy as np
import gzip
import struct

def parse_mnist(image_filename, label_filename):
    """Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded
                data.  The dimensionality of the data should be
                (num_examples x input_dim) where 'input_dim' is the full
                dimension of the data, e.g., since MNIST images are 28x28, it
                will be 784.  Values should be of type np.float32, and the data
                should be normalized to have a minimum value of 0.0 and a
                maximum value of 1.0.

            y (numpy.ndarray[dypte=np.int8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.int8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR SOLUTION
    with gzip.open(image_filename, "rb") as f:
        _, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        bytes_per_image = rows * cols
        X = np.zeros((num_images, rows * cols), dtype=np.float32)

        for i in range(num_images):
            image_data = f.read(bytes_per_image)
            if not image_data:
                break
            X[i] = np.frombuffer(image_data, dtype=np.uint8).astype(np.float32)
        X /= 255.0

    with gzip.open(label_filename, "rb") as f:
        _, num_labels = struct.unpack(">II", f.read(8))
        y = np.zeros(num_labels, dtype=np.uint8)
        bytes_per_label = 1
        
        for i in range(num_labels):
            label_data = f.read(bytes_per_label)
            if not label_data:
                break
            y[i] = np.frombuffer(label_data, dtype=np.uint8)[0]
        
    
    return X, y
    ### END YOUR SOLUTION

class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        self.X, self.y = parse_mnist(image_filename, label_filename)
        self.transforms = transforms if transforms is not None else []
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        data_point = self.X[index]
        data_point = data_point.reshape(28, 28, 1) if isinstance(index, (int, np.integer)) else data_point.reshape(-1, 28, 28, 1)
        data_point = self.apply_transforms(data_point)
        return data_point, self.y[index]
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return self.y.shape[0]
        ### END YOUR SOLUTION