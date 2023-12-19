from torch_mate.data.utils import Siamese

from test_triplet import LabelsAreData

ITERS = 1000

def test_siamese():
    data = LabelsAreData([1, 2, 3, 4], 1000)

    siamese = Siamese(data)

    # Get 100 batches
    for i, batch in enumerate(siamese):
        if i >= 1000:
            break
        
        assert len(batch) == 2, "Siamese should have 2 items"
        assert len(batch[0]) == 2, "Siamese item should have 2 data points"

        if batch[1]:
            assert batch[0][0] == batch[0][1], "Anchor and positive should be the same"
        else:
            assert batch[0][0] != batch[0][1], "Anchor and negative should be different"
