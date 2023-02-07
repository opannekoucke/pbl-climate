import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from cartopy.util import add_cyclic_point
import cartopy.crs as ccrs

from ..climate_db import ClimateDataBase

#from pydap.tool.progress import range_bar


class _Data(ClimateDataBase):
    """ Used to train wgan on a single NetCDF file

    generator return (batch_size, nlat, nlon, channels)

    For this class, all database is load in memory when normalization is processed

    """

    fields = ()

    surface_fields = ()  # List of surface fields

    codes = {}  # Codes for fields (OMM, ECMWF,..)

    def __init__(self, filename, start_date=0, stat_filename=None):

        self.filename = filename
        # Open the netcdf database
        self.ncdata = Dataset(self.filename)
        self.basename = filename[:-3]

        self.start_date = start_date  # start extraction from date 'start_date'

        # self.data = self.ncdata.variables
        if stat_filename is None:
            stat_filename = self.basename + '_stats.pkl'
        self.stat_filename = stat_filename

        super().__init__(self.ncdata.variables, self.stat_filename)

        self.lons = self.ncdata.variables['lon'][:]
        self.lats = self.ncdata.variables['lat'][:]
        self.grid_shape = (len(self.lats), len(self.lons))
        self.nlat = self.grid_shape[0]
        self.nlon = self.grid_shape[1]
        self.levels = self.ncdata.variables['lev'][:]
        self.nlev = len(self.levels)
        self.pression = {pression: index for index, pression in enumerate(self.ncdata.variables['lev'][:])}


        # Compute number of channels and the indexes for fields
        # self.channels = 6*self.nlev+1  # 6 champs 3D et 1 champ 1D
        self.index = []
        self.channels = 0
        for field in self.fields:
            if field in self.surface_fields:
                self.channels += 1
                self.index += [field]

            else:
                self.channels += self.nlev
                self.index += self.nlev*[field]
        self.index = np.asarray(self.index)
        self._loaded_data = None

    @property
    def ntimes(self):
        return len(self.data['time'])

    @property
    def _climate_idx(self, start_date=0):
        """ Compute climate over a finite period of at most 30ys """
        year = 365
        n_years = 30

        period = n_years * year  # period over which climate stats are estimated

        if self.ntimes < period + start_date:
            end_date = self.ntimes
        else:
            start_date = self.ntimes - period
            end_date = self.ntimes

        return start_date, end_date

    def __getitem__(self, k):
        """
        extract the k'th full state at a given date

        :param k: integer
        :return: the full state associated with the k^th date.
        """
        k = k % self.ntimes

        state = []
        for field in self.fields:
            data = self.data[field][k].copy()

            if field in self.surface_fields:
                # add a dimension
                data = np.expand_dims(data, axis=0)

            data = np.moveaxis(data, 0, -1)
            state.append(data)

        return np.concatenate(state, axis=-1)

    def random_generator(self, batch_size=32):
        """
        Create a random generator for PUMA by loading normalized data in memory
        :param batch_size: size of batch used
        :return: generator which construct batch of randomly chosen data

        ..Warning::
            This load all the database in memory
        """

        # -1- load normalized data in memory
        if self._loaded_data is None:
            print('Load data base in memory')
            self._loaded_data = [self(k) for k in range(self.ntimes)]

        # -2- create a database
        db = ClimateDataBase(self._loaded_data, normalized=True)

        # -3- return the generator associated with

        return db.random_generator(batch_size=batch_size)

    def plot(self, data, figsize=(12, 5), cmap='viridis', projection=ccrs.PlateCarree(), ax=None, colorbar=True):

        if projection == 'Miller':
            projection = ccrs.Miller()
        elif projection == 'EuroPP':
            projection = ccrs.EuroPP()
        else:
            projection = projection

        if ax is None:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(1, 1, 1, projection=projection)
        else:
            fig = ax.figure

        # force to periodic data.
        data = add_cyclic_point(data)
        clons = [lon for lon in self.lons] + [360.]

        CF = ax.contourf(clons, self.lats, data,
                    # transform=ccrs.PlateCarree(),
                    # cmap='nipy_spectral'
                    cmap=cmap)
        ax.coastlines()
        if colorbar:
            fig.colorbar(CF)
        # ax.set_global()
        pass



class _PUMA(object):

    fields = ('pl', 'ta', 'ua', 'va', 'wap', 'zeta', 'd', 'zg')
    surface_fields = ('pl',)

    codes = {
        'ta': 130,  # temperature
        'ua': 131, 'va': 132, 'wap': 135,  # velocity
        'zeta': 138,  # vorticity
        'pl': 152,  # log(surface pressure)
        'd': 155,  # divergence
        'zg': 156  # geopotential
    }


class _PLASIM(_PUMA):
    fields = _PUMA.fields + ('hus',)

    codes = _PUMA.codes.copy()
    codes['hus'] = 133  # specific humidity


class PUMAData(_PUMA, _Data): pass




class PLASIMData(_PLASIM, _Data): pass




class State(object):

    cmap = 'viridis'

    def __init__(self, puma, data):
        self.puma = puma
        self.data = data
        self.projection = ccrs.Miller()
        self.lats = puma.lats
        self.lons = puma.lons
        self.clons = np.asarray([lon for lon in self.lons] + [360.])
        self.levels = puma.levels
        self.nlev = len(self.levels)
        self.data_crs = ccrs.PlateCarree()

    def __getitem__(self, item):
        return self.data[:, :, self.puma.index == item]

    def __getattr__(self, item):
        try:
            return getattr(self.puma, item)
        except:
            raise

    def plot_cross_section(self, field, lat, lons=None, minmax=None, figsize=(12,5), title=None,ax=None,
                            colorbar=True):

        title = title if title is not None else f" {field} at latitude {self.lats[lat]}"

        if ax is None:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111)
        else:
            fig = ax.figure

        data = self[field].copy()
        data = data[lat, :, :]
        # CS = plt.contourf(self.lons, self.levels[::-1], data)
        if lons is not None:
            CS = ax.contourf(self.lons[lons[0]:lons[1]], range(self.nlev),
                              data[lons[0]:lons[1], ::-1].T)
        else:
            CS = ax.contourf(data[:, ::-1].T)
        if colorbar:
            fig.colorbar(CS)
        plt.title(title)

    def _set_default(self, kwargs):
        if 'ax' not in kwargs:
            kwargs['ax']=None
        return kwargs

    def _set_plot(self, **kwargs):

        kwargs = self._set_default(kwargs)

        if kwargs['ax'] is None:
            #fig = plt.figure(figsize=(12, 5))
            #ax = fig.add_subplot(1, 1, 1, projection=self.projection)
            ax = plt.axes(projection=self.projection)
            fig = ax.figure
        else:
            ax = kwargs['ax']
            fig = ax.figure

        return ax, fig

    def figure(self,*args,**kwargs):
        self._figure = plt.figure(*args,**kwargs)

    def add_subplot(self,*args, projection=None):
        if projection is None:
            projection = self.projection
        ax = self._figure.add_subplot(*args, projection=projection)
        return ax


    def plot_field(self, field, **kwargs):
        """
        Plot `field` at a specified level (if level is given)

        Call signatures:
            plot_field(field, level=0, **kwargs)

        :param field:
        :param level:
        :param kwargs:
        :return:
        """
        try:
            level = kwargs.pop('level')
        except:
            level = 0

        if 'minmax' in kwargs:
            minmax = kwargs.pop('minmax')
            if minmax is not None:
                vmin = minmax[0]
                vmax = minmax[1]
        else:
            vmin = None
            vmax = None

        data = self[field][:,:,level]
        data = add_cyclic_point(data)

        ax, fig = self._set_plot(**kwargs)
        #the following lines replace: `self.puma.plot(data, **kwargs)`
        CB = ax.contourf(self.clons, self.lats, data, cmap=self.cmap,
                         transform=self.data_crs,
                         vmin=vmin,vmax=vmax)
        ax.coastlines()

        try:
            if kwargs['colorbar']:
                ax.figure.colorbar(CB)
        except:
            pass
        return ax

    def plot_wind(self, level, **kwargs):

        ax, fig = self._set_plot(**kwargs)

        # Get data
        u, v = self['ua'], self['va']
        u = u[:, :, level]
        v = v[:, :, level]

        # Set to plot
        u = add_cyclic_point(u)
        v = add_cyclic_point(v)

        ax.quiver(self.clons, self.lats, u, v, transform=self.data_crs)
        ax.coastlines()
        return ax

    def plot_geopot(self,level, **kwargs):

        # Handle ax if present else
        ax, fig = self._set_plot(**kwargs)
        kwargs['ax'] = ax
        kwargs['level'] = level

        self.plot_field('zg',**kwargs)
        self.plot_wind(**kwargs)
        plt.title(f'zg at {self.levels[level]}')
        return ax

