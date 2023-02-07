from unittest import TestCase
from pydap.data.climate_db import ClimateDataBase
import numpy as np
from numpy.linalg import norm

class TestClimateDataBase(TestCase):

    def test_homogeneous_data(self):
        # Create homogeneous data
        data = np.random.normal(size=(30, 5, 10))

        # Create the database
        db = ClimateDataBase(data)

        # Compute the mean/std using numpy
        data_MEAN = data.mean(axis=0)
        data_STD = data.std(axis=0)

        # Validation of both MEAN & STD
        self.assertTrue(norm(data_MEAN-db.MEAN)==0.)
        self.assertTrue(norm(data_STD - db.STD)==0.)

        # Validation of the mean/std of the centered data
        cdb = ClimateDataBase([db(k) for k in db.climate_range])

        self.assertTrue(norm(cdb.MEAN)<1e-4)
        self.assertTrue(norm(cdb.STD-1)<1e-4)


    def test_heterogeneous_data(self):
        # Create heterogeneous data
        data = np.random.normal(size=(2, 30, 5, 10, 5))
        data[0] = 1 + data[0] * 1
        data[1] = 1168168435138468441 + data[1] * 14843513578761351387684611554
        data = np.moveaxis(data, 0, -1)

        # Create the database
        db = ClimateDataBase(data)

        # Compute the mean/std using numpy
        data_MEAN = data.mean(axis=0)
        data_STD = data.std(axis=0)

        # Validation of both MEAN & STD
        self.assertTrue(norm(data_MEAN-db.MEAN)==0.)
        self.assertTrue(norm(data_STD - db.STD)==0.)

        # Validation of the mean/std of the centered data
        cdb = ClimateDataBase([db(k) for k in db.climate_range])

        self.assertTrue(norm(cdb.MEAN)<1e-4)
        self.assertTrue(norm(cdb.STD-1)<1e-4)