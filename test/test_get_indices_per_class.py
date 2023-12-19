from torch_mate.data.utils.get_indices_per_class import get_indices_per_class

from utils import LabelsAreData

SAMPLES_PER_CLASS = 100
LABELS = [0,1,2,3,4,5]
LENGTH = len(LABELS) * SAMPLES_PER_CLASS

def test_get_indices_per_class():
    data = LabelsAreData(LABELS, LENGTH)

    expected = {}

    for label in LABELS:
        expected[label] = [i for i in range(label, LENGTH, len(LABELS))]

    assert get_indices_per_class(data) == expected
    assert get_indices_per_class(data, samples_per_class=-SAMPLES_PER_CLASS) == expected

    data = LabelsAreData(LABELS, LENGTH, transposed=True)

    expected = {}

    for label in LABELS:
        expected[label] = [i for i in range(label*SAMPLES_PER_CLASS, (label+1)*SAMPLES_PER_CLASS)]

    assert get_indices_per_class(data) == expected
    assert get_indices_per_class(data, samples_per_class=SAMPLES_PER_CLASS) == expected
