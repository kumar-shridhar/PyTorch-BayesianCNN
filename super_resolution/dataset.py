import torch.utils.data as data

from os import listdir
from os.path import join
from PIL import Image


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def load_img(filepath):
    img = Image.open(filepath).convert('YCbCr')
    y, cb, cr = img.split()
    return y, cb, cr


class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, input_transform=None, target_transform=None):
        super(DatasetFromFolder, self).__init__()
        self.image_filenames = [join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)]

        self.input_transform = input_transform
        self.target_transform = target_transform

    def getitem_full(self, index):
        y, cb, cr = load_img(self.image_filenames[index])
        target = y.copy()
        if self.input_transform:
            input = self.input_transform(y)
        if self.target_transform:
            target = self.target_transform(target)
            # cb = self.target_transform(cb)
            # cr = self.target_transform(cr)

        return input, target, cb, cr

    def __getitem__(self, index):
        input, _, _ = load_img(self.image_filenames[index])
        target = input.copy()
        if self.input_transform:
            input = self.input_transform(input)
        if self.target_transform:
            target = self.target_transform(target)

        return input, target

    def __len__(self):
        return len(self.image_filenames)
