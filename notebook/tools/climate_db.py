import random
import pickle
import logging
import numpy as np
#from pydap.util.stats import compute_mean, compute_std

def compute_mean(data):
    """
    Compute the mean of data
    :param data: can be a structure or a generator
    :return:

    .. example::
    >>> n = 100
    >>> data = list(range(n))
    >>> compute_mean(data)
    49.5

    """
    mean = 0
    size = 0
    for elm in data:
        mean = mean + elm
        size += 1
    mean /= size
    return mean


def compute_std(data, mean):
    """
    Compute the std of `data` knowing the `mean`
    :param data: structure or generator
    :param mean: mean of `data`
    :return:
    """
    std = 0
    size = 0
    for elm in data:
        std = std + (elm - mean)**2
        size += 1
    std /= size
    return sqrt(std)


class ClimateDataBase(object):
    """ Generical structure to handle climate data """

    def __init__(self, data, filename=None, MEAN=None, STD=None, normalized=False):
        self.data = data
        self.normalized = normalized

        if not self.normalized:

            self.MEAN = MEAN
            self.STD = STD

            if filename is not None:
                try:
                    self._load_clim_statistics(filename)
                except FileNotFoundError:
                    self._compute_statistics()
                    self._save_clim_statistics(filename)
            else:
                self._compute_statistics()

            self._normalize()

    def _compute_statistics(self):
        logging.debug('Compute the climate for indexed state {self._climate_idx}')
        logging.debug('Start computation of the means')
        if self.MEAN is None:
            self.MEAN = compute_mean(self._climate_states)
        logging.debug('Start computation of the std')
        if self.STD is None:
            self.STD = compute_std(self._climate_states, self.MEAN)

    def _normalize(self):
        self._iSTD = self.STD.copy()
        self._iSTD[self.STD==0] = 1.
        self._iSTD = 1/self._iSTD

    @property
    def ntimes(self):
        """ Length of the climate data base"""
        return len(self.data)

    @property
    def _climate_states(self):
        """ Return the generator over states used in the computation of the climate """
        return (self[k] for k in self.climate_range)

    @property
    def _climate_idx(self):
        start_date = 0
        end_date = self.ntimes-1
        return start_date, end_date

    @property
    def climate_range(self):
        start_date,  end_date = self._climate_idx
        return range(start_date, end_date)

    def __getitem__(self, item):
        return self.data[item]

    def __call__(self, k):
        """ return de k'th state with applying the normalization if `self.normalized` is `True` """
        data = self[k]
        if not self.normalized:
            data = data - self.MEAN
            data = data*self._iSTD
        return data

    def rescale(self, data):
        """ Convert normalized data into physical values """
        return self.MEAN + self.STD*data

    def __iter__(self):
        self._iter = 0
        self._stop = self.ntimes
        return self

    def __next__(self):
        if self._iter < self._stop:
            self._iter += 1
            return self[self._iter-1]
        else:
            del self._iter, self._stop
            raise StopIteration()

    def _save_clim_statistics(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump([self.MEAN, self.STD], file)

    def _load_clim_statistics(self, filename):
        with open(filename, 'rb') as file:
            self.MEAN, self.STD = pickle.load(file)

    def random_sample(self, batch_size=32, start_date=0, end_date=None, idx=None):
        """ Create a random sample of date """
        # Extract data from PUMA so the output is (batch_size, nlat, nlon, 51)
        # min_date is the minimal selected date: this can be used to only extract data after the first year ..

        if end_date is None:
            end_date = self.ntimes-1

        # 1. Sample indexes of selected data: sample start from date `self.start_date`
        #    Note: using random.randint is compatible with multiprocessing
        #    (not the same random number on two processes)
        if idx is None:
            idx = []
            for _ in range(batch_size):
                idx.append(random.randint(start_date, end_date))
        #idx = np.random.randint(self.start_date, self.ntimes, size=batch_size)

        # 2. Extract data from netcdf file (with normalization if `self.normalize=True`)
        states = np.concatenate([self(k)[None, :] for k in idx], axis=0)

        return states

    def random_generator(self, batch_size=32):
        """ Generator that yields a random sample """
        while True:
            yield self.random_sample(batch_size=batch_size)
