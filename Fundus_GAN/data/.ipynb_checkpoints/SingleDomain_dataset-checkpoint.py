import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import util.util as util

########################################## 添加的内容 ##############################################
class SingleDomainDataset(BaseDataset):
    """
    This dataset class can load a single domain dataset.

    It requires one directory to host images from domain A '/path/to/data/testA'.
    You can test the model with the dataset flag '--dataroot /path/to/data'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/testA'

        if opt.phase == "test" and not os.path.exists(self.dir_A) \
           and os.path.exists(os.path.join(opt.dataroot, "valA")):
            self.dir_A = os.path.join(opt.dataroot, "valA")

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/testA'
        self.A_size = len(self.A_paths)  # get the size of dataset A

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, A_paths
            A (tensor)       -- an image in the input domain
            A_paths (str)    -- image paths
        """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within the range
        A_img = Image.open(A_path).convert('RGB')

        # Apply image transformation
        transform = get_transform(self.opt)
        A = transform(A_img)

        return {'A': A, 'A_paths': A_path}

    def __len__(self):
        """Return the total number of images in the dataset.
        """
        return self.A_size
########################################## 添加的内容 ##############################################