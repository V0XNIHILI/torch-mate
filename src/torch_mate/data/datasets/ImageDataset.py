import os

from torch.utils.data import Dataset

from PIL import Image

class ImageDataset(Dataset):
    def __init__(self, data_dir: str, min_samples = 10, transform=None):
        """A torch Dataset for image classification. This dataset assumes that every class has its own subdirectory
        in the data_dir directory. The class names are inferred from the subdirectory names.

        Args:
            data_dir (str): Path to the directory containing the data.
            min_samples (int, optional): Minimum number of samples in a class to include this class in the dataset. Defaults to 10.
            transform (Callable[[torch.Tensor], torch.Tensor], optional): Transform to apply to images. Defaults to None.
        """

        self.data_dir = data_dir
        self.transform = transform

        self.data = []
        self.labels = []
        self.classes = []

        class_names = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

        i = 0

        for class_name in class_names:
            class_dir = os.path.join(self.data_dir, class_name)

            files_per_class = []
            labels_per_class = []

            for file_name in os.listdir(class_dir):
                if file_name.endswith(".png"):
                    file_path = os.path.join(class_dir, file_name)
                    files_per_class.append(file_path)
                    labels_per_class.append(i)

            if len(files_per_class) > min_samples:
                self.data.extend(files_per_class)
                self.labels.extend(labels_per_class)
                self.classes.append(class_name)

                i += 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        img = Image.open(self.data[idx])
        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)

        return img, label