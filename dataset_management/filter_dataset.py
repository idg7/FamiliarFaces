import os
import numpy as np
import shutil
import glob
import const


class BootstrapFilterDatasetSetup(object):
    """
    A dataset setup, filtering ids and saving them in a new dir

    Filter:
    - Search for [n_ids] IDs with max([n_trains]) + [n_test] + [n_trains] data points,
    - and filter out all IDs in [exclude_ids]. then save at the dataset_root_dir by specific format:

    {dataset_root_dir}/{dataset}/subset-{n_ids}-{max(n_trains)}-{n_val}-{n_test}/{n_train}/

    Inside there will be train, test, val as expected.
    """
    def __init__(self, dataset_root_dir, n_trains, n_val, n_test, n_ids, dataset_fold_name=None, exclude_ids=[]):
        self.__dataset_root_dir = dataset_root_dir
        self.__n_trains = n_trains
        self.__n_val = n_val
        self.__n_test = n_test
        self.__n_ids = n_ids
        self.__dataset_fold_name = dataset_fold_name
        self.__exclude_ids = exclude_ids

    def setup_dataset(self, dataset, data_loc):

        dataset_dirs = []

        good_ids = self.__get_good_ids(data_loc, np.max(self.__n_trains) + self.__n_val + self.__n_test)
        if len(good_ids) < self.__n_ids:
            raise ValueError(f'parameters produced only {len(good_ids)} of {self.__n_ids} desired identities. reduce number of images/ID required')

        if self.__dataset_fold_name is None:
            self.__dataset_fold_name == 'dataset'

        new_dir = self.__create_dir(dataset)
        if os.path.exists(new_dir):
            shutil.rmtree(new_dir)

        for n_train in self.__n_trains:
            dataset_dirs.append(self.__segment_dataset(good_ids, new_dir, n_train))

    def __get_dir(self, dataset):
        return os.join(self.__dataset_root_dir, f'{dataset}', f'subset-{self.__n_ids}-{np.max(self.__n_trains)}-{self.__n_val}-{self.__n_test}')

    def __get_good_ids(self, data_loc, n_items):
        """
        Gets all ids with sufficient data points, after excluding bad ids
        :param data_loc: Path to data
        :param n_items: Number of items in dataset
        :param dataset_fold_name: The name of the fold
        :param exclude_ids: IDs to exclude from the list
        :return: List of good IDs
        """
        good_ids = []
        for i, id_fold in enumerate(glob.glob(os.path.join(data_loc, self.__dataset_fold_name, '*'))):
            if len(glob.glob(os.join(id_fold, '*'))) >= n_items:
                if os.path.basename(id_fold) not in self.__exclude_ids:
                    good_ids.append(id_fold)
        return good_ids

    def __segment_dataset(self, good_ids, dest_dir, n_train):
        """
        Segmenting and saving the dataset (train, test, val) of the good_ids based into dest_dir, given the number of data_points given per segment
        :param good_ids: list of good ids to use
        :param dest_dir: destination directory for the dataset
        :param n_train: number of training examples per ID
        :param n_val: number of validation examples per ID
        :param n_test: number of testing examples per ID
        :param n_ids: number of IDs to use
        :return: the directory containing the segmented dataset (to train, val, test)
        """
        np.random.seed(const.SEED)
        permuted_ids = np.random.permutation(good_ids)

        segmented_dataset_dir = os.path.join(dest_dir, f'train-{n_train}')

        for i in range(self.__n_ids):
            id_fold = permuted_ids[i]

            id_dir_train = os.path.join(segmented_dataset_dir, 'train', os.path.basename(id_fold))
            id_dir_val = os.path.join(segmented_dataset_dir, 'val', os.path.basename(id_fold))
            id_dir_test = os.path.join(segmented_dataset_dir, 'test', os.path.basename(id_fold))
            os.makedirs(id_dir_val, exist_ok=True)
            os.makedirs(id_dir_train, exist_ok=True)
            os.makedirs(id_dir_test, exist_ok=True)
            all_ims = sorted(glob.glob(id_fold + '/*'))
            for im_i, im in enumerate(all_ims):
                if im_i < self.__n_test:
                    os.symlink(im,os.path.join(id_dir_test,os.path.basename(im)))
                elif im_i < self.__n_test+self.__n_val:
                    os.symlink(im, os.path.join(id_dir_val,os.path.basename(im)))
                elif im_i < n_train+self.__n_val+self.__n_test:
                    os.symlink(im, os.path.join(id_dir_train,os.path.basename(im)))
                else:
                    break

        return segmented_dataset_dir
