import os
import glob

def verify_dataset(path, min_classes, min_items_train, min_items_val, min_items_test, min_total_train, min_total_val, min_total_test):
    pass

def verify_segment(path, min_classes=2, min_items_per_class=1, min_total_examples=2):
    """
    Checking if the given dataset contains enough data for the needed assignment
    To be used on specific data segment (train\val\test)
    :param path: Path to the dataset segment
    :param min_classes: Minimum compatible classes to train\test\validate over. Default=2.
    :param min_items_per_class: The minimum number of items per class we wish to train\test\val. Default 1.
    :param min_total_examples: Minimum total examples (per entire segment). default 2
    :return: boolean stating if the dataset is compatible for the assignment
    """
    glob_search_all_filter = '*'
    classes = glob.glob(os.path.join(path, glob_search_all_filter))
    num_classes = len(classes)

    if num_classes < min_classes:
        # Not enough classes
        return False

    total_items = 0
    for cl in classes:
        items = glob.glob(os.path.join(cl, glob_search_all_filter))
        cl_items = len(items)

        if cl_items < min_items_per_class:
            # Not enough items in a class
            return False

        total_items += len(items)

    if total_items < min_total_examples:
        # Not enough items in total
        return False

    # All's good
    return True
