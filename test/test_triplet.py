from torch_mate.data.utils import Triplet

from utils import LabelsAreData

ITERS = 1000

def test_triplet():
    data = LabelsAreData([1, 2, 3, 4], 1000)

    triplet = Triplet(data)

    # Get 100 batches
    for i, batch in enumerate(triplet):
        if i >= ITERS:
            break
        
        assert len(batch) == 2, "Triplet should have 2 items"
        assert len(batch[0]) == 3, "Triplet item should have 3 data points"
        assert batch[0][0] == batch[1], "Anchor and label should be the same"
        assert batch[0][1] == batch[0][1], "Anchor and positive should be the same"
        assert batch[0][2] != batch[0][1], "Anchor and negative should be different"
