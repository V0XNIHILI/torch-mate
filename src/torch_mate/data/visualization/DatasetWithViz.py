from torch.utils.data import Dataset


class DatasetWithViz(Dataset):

    def show(self, *args) -> None:
        raise NotImplementedError
