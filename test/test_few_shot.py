from collections import Counter

from torch_mate.data.utils import FewShot

from test_triplet import LabelsAreData

ITERS = 1000
N_WAY = 5
K_SHOT = 2
K_QUERY = 3

def test_few_shot():
    data = LabelsAreData(list(range(100)), 1000)

    for always_include_classes in [None, [43,67]]:
        few_shot = FewShot(data, N_WAY, K_SHOT, K_QUERY, always_include_classes=always_include_classes)

        always_include_class_mapping = None

        # Get 100 batches
        for i, batch in enumerate(few_shot):
            if i >= 1000:
                break

            ((X_train, X_test), (y_train, y_test)) = batch

            assert len(X_train) == N_WAY * K_SHOT, "X_train should have N_WAY * K_SHOT items"
            assert len(X_test) == N_WAY * K_QUERY, "X_test should have N_WAY * K_QUERY items"
            assert len(y_train) == N_WAY * K_SHOT, "y_train should have N_WAY * K_SHOT items"
            assert len(y_test) == N_WAY * K_QUERY, "y_test should have N_WAY * K_QUERY items"

            assert len(set(y_train.tolist())) == N_WAY, "y_train should have N_WAY unique items"
            assert len(set(y_test.tolist())) == N_WAY, "y_test should have N_WAY unique items"
            assert len(set(X_train.tolist())) == N_WAY, "X_train should have N_WAY unique items"
            assert len(set(X_test.tolist())) == N_WAY, "X_test should have N_WAY unique items"

            # Check that each unique item has the same number of occurrences
            assert all([y_train.tolist().count(i) == K_SHOT for i in range(N_WAY)]), "y_train should have K_SHOT items for each unique class (N_WAY total)"
            assert all([y_test.tolist().count(i) == K_QUERY for i in range(N_WAY)]), "y_test should have K_QUERY items for each unique class (N_WAY total)"

            counts_per_class_train = Counter(map(lambda x: x.item(), X_train))
            counts_per_class_test = Counter(map(lambda x: x.item(), X_test))

            assert all([count == K_SHOT for count in counts_per_class_train.values()])
            assert all([count == K_QUERY for count in counts_per_class_test.values()])

            train_sample_label_mapping = set([(x.item(), y.item()) for x, y in zip(X_train, y_train)])
            test_sample_label_mapping = set([(x.item(), y.item()) for x, y in zip(X_test, y_test)])

            assert train_sample_label_mapping == test_sample_label_mapping, "X_train and X_test should have the same label mapping"

            if always_include_classes is not None:
                if always_include_class_mapping is None:
                    always_include_class_mapping = dict(train_sample_label_mapping)
                    always_include_class_mapping = {key: always_include_class_mapping[key] for key in always_include_classes}
                else:
                    new_always_include_class_mapping = dict(train_sample_label_mapping)
                    new_always_include_class_mapping = {key: always_include_class_mapping[key] for key in always_include_classes}

                    assert new_always_include_class_mapping == always_include_class_mapping, "Always include class mapping should always have the same few-shot labels"

            assert list(set(y_train.tolist())) == list(range(N_WAY)), "y_train should have labels 0..N_WAY-1"
            assert list(set(y_test.tolist())) == list(range(N_WAY)), "y_test should have labels 0..N_WAY-1"

            assert len(set(y_train.tolist()) & set(y_test.tolist())) == N_WAY, "y_train and y_test should have the same items"
            assert set(X_train.tolist()) == set(X_test.tolist()), "X_train and X_test should have the same items"

            if always_include_classes is not None:
                assert all(ele in X_train for ele in always_include_classes), "Every class in the always_include_classes should appear in the X_train samples"
                assert all(ele in X_test for ele in always_include_classes), "Every class in the always_include_classes should appear in the X_test samples"


if __name__ == "__main__":
    test_few_shot()