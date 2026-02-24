"""
This module contains five classes to load and analize the ELFIN data. The five classes are:
1. State --- Load the ELFIN ephemeris,
2. EPD --- load the ELFIN L1 and L2 cdf files,
3. MagEphem  --- Use IRBEM to produce magnetic ephemeris,
4. EPD_PAD --- Use the L1 and L2 EPD data and make plots including omnidirectional and BLC/DLC flux. 
5. EPD_PAD_ARTEMYEV --- Same as EPD_PAD, but for the custom files provided by Dr. Anton Artemyev.

This module also contains two helper functions, download_all which downloads all of the available L1, 
and L2 EPD data, as well as the state files.
"""

from datetime import datetime, timedelta, timezone
import pathlib
from collections import namedtuple
import dateutil.parser
from typing import Tuple
import warnings

import pandas as pd
import numpy as np
import xarray as xr
import scipy.signal
import IRBEM
import matplotlib
import matplotlib.colors
import matplotlib.pyplot as plt
from matplotlib.dates import date2num
import aacgmv2
import manylabels
import cdflib
import cdasws
from asilib.download import Downloader

import elfinasi
from .validation import validate_time_range

Re = 6378.14  # km

base_url = 'https://data.elfin.ucla.edu/'
local_base_dir = pathlib.Path.home() / 'elfin-data'
if not local_base_dir.exists():
    local_base_dir.mkdir()
    print(f'Made ELFIN data directory in {local_base_dir}')


class State:
    def __init__(self, sc_id, day, load=True) -> None:
        """
        Loads the ELFIN ephemeris data. Access variables how you
        access dictionary values.

        Parameters
        ----------
        sc_id: str
            The spacecraft id, either "A" or "B". Case insensitive.
        day: str or datetime.datetime
            The date to load.
        load: bool
            If True, download and load the state data, otherwise just
            download it.

        Methods
        -------
        keys()
            Get a list of variable names.

        Attributes
        ----------
        file_path: pathlib.Path
            The full path to the CDF State file.
        
        Example
        -------
        sc_id = 'A'
        day = '2020-01-01'
        elfin_state = State(sc_id, day)
        print(elfin_state.keys())
        print(elfin_state['epoch'])  # 'time' works as well.
        print(elfin_state['ela_pos_gei'])
        """
        if isinstance(day, str):
            day = dateutil.parser.parse(day)
        self.day = day
        self.sc_id = sc_id
        self.file_pattern = (
            f'el{self.sc_id.lower()}_l1_state_defn_{self.day.strftime("%Y%m%d")}_v02.cdf'
            )
        if load:
            self.load()
        else:
            self.download()
        pass
    
    def load(self):
        # Try to find the file locally first
        self._find_local_file()
        if not hasattr(self, 'file_path'):
            self.download()
        self.state_obj = cdflib.CDF(self.file_path)
        return self.state_obj
    
    def _find_local_file(self):
        """
        Tries to find the local file.
        """
        file_paths = list(local_base_dir.rglob(self.file_pattern))
        if len(file_paths) == 1:
            self.file_path = file_paths[0]
        return
    
    def download(self):
        # Don't download a file if it is already downloaded
        self._find_local_file()
        if hasattr(self, 'file_path'):
            return self.file_path
        
        _downloader = Downloader(
            base_url + f'el{self.sc_id.lower()}/l1/state/defn/{self.day.year}/',
            local_base_dir / f'el{self.sc_id.lower()}' / 'l1' / 'state' / 'defn' / f'{self.day.year}'
            )
        _matched_downloaders = _downloader.ls(self.file_pattern)
        if len(_matched_downloaders) == 0:
            raise FileNotFoundError(
                f"No ELFIN-{self.sc_id.upper()} state files found at {local_base_dir} "
                f"(and subdirectories) or {_downloader.url} that match : {self.file_pattern}."
                )
        self.file_path = _matched_downloaders[0].download()
        return self.file_path
    
    def keys(self):
        if hasattr(self.state_obj.cdf_info(), 'zVariables'):
            return self.state_obj.cdf_info().zVariables
        return self.state_obj.cdf_info()['zVariables']
    
    def __getitem__(self, _slice):
        """
        Access the cdf variables using keys (i.e., ['key']).
        """
        if isinstance(_slice, str):
            if ("epoch" in _slice.lower()) or ("time" in _slice.lower()):
                if hasattr(self, 'epoch'):
                    return self.epoch
                self.epoch = np.array(cdflib.cdfepoch.to_datetime(
                    self.state_obj.varget(f'el{self.sc_id.lower()}_state_time')
                    ))
                return self.epoch
            else:
                try:
                    return self.state_obj[_slice]
                except KeyError:
                    raise KeyError(f'{_slice} is unrecognized. Try one'
                                   f' of these: {self.keys()}')
        else:
            raise IndexError('Only slicing with string keys is supported.')


class EPD:
    def __init__(self, sc_id, day, load=True, level=2, species='electron') -> None:
        """
        Loads the ELFIN energetic particle data. Access variables how you
        access dictionary values.

        Parameters
        ----------
        sc_id: str
            The spacecraft id, either "A" or "B". Case insensitive.
        day: str or datetime.datetime
            The date to load.
        load: bool
            If True, download and load the EPD data, otherwise just
            download it.
        level: int
            The data level, can be either 1 (spin-sector), or 2 (pitch angle) data.
        species: str
            The particle species. Can be either "electron" or "ion".

        Methods
        -------
        keys()
            Get a list of variable names.

        Attributes
        ----------
        file_path: pathlib.Path
            The full path to the CDF EPD file.
        
        Example
        -------
        # Load a L2 flux file.

        import matplotlib.pyplot as plt
        import matplotlib.colors
        from datetime import datetime

        from pad import EPD

        energy = 98  # keV

        sc_id = 'B'.lower()
        day = '2020-09-02'
        elfin_epd = EPD(sc_id, day)
        print(elfin_epd.keys())

        energy_channels = np.round(elfin_epd[f'el{sc_id}_pef_energies_mean']).astype(int)
        ide = np.where(energy_channels == energy)[0]
        assert len(ide) == 1, f'{energy=} keV is an invalid ELFIN-EPD energy channel. Try one of these: {energy_channels}'
        ide = ide[0]

        fig, ax = plt.subplots(2, 1, sharex=True, figsize=(6, 6))
        _zipped = zip(
            elfin_epd[f'el{sc_id}_pef_energies_min'], 
            elfin_epd[f'el{sc_id}_pef_energies_max'], 
            elfin_epd[f'el{sc_id}_pef_Et_nflux'].T
            )
        for low_E, high_E, flux in _zipped:
            ax[0].plot(elfin_epd[f'el{sc_id}_pef_et_time'], flux, label=f'{round(low_E)}-{round(high_E)} keV')

        # We must do this to make the time array compatible with the elX_pef_fs_epa_spec and 
        # elX_pef_fs_Epat_nflux arrays.
        _spec_times = np.broadcast_to(
            elfin_epd[f'el{sc_id}_pef_fs_time'], elfin_epd[f'el{sc_id}_pef_fs_epa_spec'].T.shape
            ).T

        p1 = ax[1].pcolormesh(
            _spec_times, 
            elfin_epd[f'el{sc_id}_pef_fs_epa_spec'], 
            elfin_epd[f'el{sc_id}_pef_fs_Epat_nflux'][..., ide],
            norm=matplotlib.colors.LogNorm(),
            shading='auto'
            )
        plt.colorbar(
            p1, 
            orientation='horizontal', 
            label=f'Electron flux [{elfin_epd.attrs(f"el{sc_id}_pef_fs_Epat_nflux")["UNITS"]}]'
            )

        ax[0].set(
            yscale='log', title=f'ELFIN-{sc_id.upper()} | {day}', 
            ylabel=elfin_epd.attrs(f"el{sc_id}_pef_Et_nflux")["UNITS"]
            )
        ax[0].legend(fontsize=6)
        ax[1].set(ylabel='Pitch angle [deg]')
        plt.tight_layout()
        plt.show()
        """
        if isinstance(day, str):
            day = dateutil.parser.parse(day)
        self.day = day
        self.sc_id = sc_id
        self.level = level
        self.species = species.lower()
        
        assert self.species in ['electrons', 'electron', 'ions', 'ion'], (
            f'ELFIN-EPD {self.species} data product is unsupported. Try either '
            f'"electron" or "ion".'
            )
        if self.species[-1] == 's':
            self.species = self.species[:-1]

        assert level in [1, 2], f'ELFIN EPD data {level=} is unsupported.'
        self.file_pattern = (
            f'el{self.sc_id.lower()}_l{self.level}_epd{self.species[0]}f_'
            f'{self.day.strftime("%Y%m%d")}_v*.cdf'
            )
        if load:
            self.load()
        else:
            self.download()
        pass
    
    def load(self):
        # Try to find the file locally first
        self._find_local_file()
        if not hasattr(self, 'file_path'):
            self.download()
        self.epd_obj = cdflib.CDF(str(self.file_path))
        return self.epd_obj
    
    def _find_local_file(self):
        """
        Tries to find the local file.
        """
        file_paths = list(local_base_dir.rglob(self.file_pattern))

        if len(file_paths) == 0:
            # Check if there is a symlink and follow it to the source folder. I need this since
            # the L2 files are stored on Box (may no longer be necessary?).
            symlink_dir = local_base_dir / f'el{self.sc_id.lower()}' / f'l{self.level}' / 'epd'
            if symlink_dir.is_symlink():
                _symlink_target_dir = symlink_dir.readlink()
                file_paths = list(_symlink_target_dir.rglob(self.file_pattern))
            else:
                return  # not a symlink so the file does not exist.

        if len(file_paths) == 1:
            self.file_path = file_paths[0]
        elif len(file_paths) > 1:
            raise FileExistsError(
                f'{len(file_paths)} files found, but only one is expected. {file_paths=}.'
                )
        else:
            raise FileNotFoundError(
                f'No level {self.level} ELFIN-{self.sc_id.upper()} {self.species} EPD '
                f'files found in the {local_base_dir} top-level directory that match '
                f': {self.file_pattern}.'
                )
        return
    
    def download(self):        
        _downloader = Downloader(
            base_url + f'el{self.sc_id.lower()}/l{self.level}/epd/fast/{self.species}/{self.day.year}/',
            (local_base_dir / f'el{self.sc_id.lower()}' / f'l{self.level}' / 'epd' / 
                'fast' / self.species / f'{self.day.year}')
            )
        _matched_downloaders = _downloader.ls(self.file_pattern)
        if len(_matched_downloaders) == 0:
            raise FileNotFoundError(
                f"No level {self.level} ELFIN-{self.sc_id.upper()} {self.species} "
                f"EPD files found at {local_base_dir} (and subdirectories) or "
                f"{_downloader.url} that match : {self.file_pattern}."
                )
        self.file_path = _matched_downloaders[0].download()
        return self.file_path
    
    def keys(self):
        return self.epd_obj.cdf_info()['zVariables']
    
    def attrs(self, var):
        return self.epd_obj.varattsget(var)
    
    def __getitem__(self, _slice):
        """
        Access the cdf variables using keys (i.e., ['key']).
        """
        if isinstance(_slice, str):
            if ("epoch" in _slice.lower()) or ("time" in _slice.lower()):
                if self.level == 2:
                    try:
                        self.epoch = np.array(cdflib.cdfepoch.to_datetime(
                            self.epd_obj.varget(_slice.lower())
                            ))
                    except (KeyError, ValueError) as err:
                        _time_keys = [key for key in self.keys() if 'time' in key]
                        raise ValueError(
                            f'{_slice.lower()} is not a valid time variable. '
                            f'Try one of these: {_time_keys}'
                            ) from err

                elif self.level == 1:
                    if hasattr(self, 'epoch'):
                        return self.epoch
                    self.epoch = np.array(cdflib.cdfepoch.to_datetime(
                        self.epd_obj.varget(f'el{self.sc_id.lower()}_p{self.species[0]}f_time')
                        ))
                return self.epoch
            else:
                try:
                    return self.epd_obj[_slice]
                except KeyError:
                    raise KeyError(f'{_slice} is unrecognized. Try one'
                                   f' of these: {self.keys()}')
        else:
            raise IndexError('Only slicing with integer keys is supported.')


class MagEphem:
    def __init__(self, sc_id, day, overwrite=False, t89=False) -> None:
        """
        Load (and calculate if necessary) ELFIN's L-shell and MLT using the IGRF with or without the
        T89 magnetic field model. If calculating the magnetic field coordinates, the data is saved to hdf5 files
        with the same name as the state files, with the exception that the word "state" is replaced
        by "magephem".

        The magnetic ephemeris variables can be accessed like a dictionary, including the .keys() 
        method to list all of the keys.
        """
        self.sc_id = sc_id
        self.day = day
        self.overwrite = overwrite
        self.t89 = t89
        self.state = State(self.sc_id, self.day)

        self.file_path = pathlib.Path(
            str(self.state.file_path).replace('state', 'magephem')
            )
        self.file_path = self.file_path.parent / self.file_path.name.replace('cdf', 'h5')
        if not self.file_path.parent.exists():
            self.file_path.parent.mkdir(parents=True)
        self.generate_magnetic_coordinates()
        return
    
    def __getitem__(self, _slice):
        """
        Access the cdf variables using keys (i.e., ['key']).
        """
        if isinstance(_slice, str):
            if ("epoch" in _slice.lower()) or ("time" in _slice.lower()):
                return self.data['epoch']
            else:
                try:
                    # Check the transformed_state variable first and then reference the hdf5 file.
                    return self.data[_slice]
                except KeyError:
                    raise KeyError(f'{_slice} is unrecognized. Try one'
                                   f' of these: {self.keys()}')
        else:
            raise IndexError('Only slicing with integer keys is supported.')
    
    def generate_magnetic_coordinates(self):
        """
        Run the magnetic field model.
        """
        if self.t89:
            kext = 'T89'
        else:
            kext = 'None'
        self.model = IRBEM.MagFields(kext=kext, sysaxes=5)
        self.magephem = np.full((self.state['epoch'].shape[0], 2), np.nan)

        X = {f'x{j}': pos_j/Re for j, pos_j in enumerate(self.state[f'el{self.sc_id}_pos_gei'].T, start=1)}
        X['time'] = [dateutil.parser.parse(str(ti)) for ti in self.state['epoch']]
        if self.t89:
            maginput = {'Kp': self._get_kp(self.state['epoch'])}
        else:
            maginput = {}
        _magephem_dict = self.model.make_lstar(X, maginput=maginput)
        
        Lm = np.abs(np.array(_magephem_dict['Lm']))
        MLT = np.array(_magephem_dict['MLT'])
        Lm[Lm == 1E31] = np.nan

        _coords_obj = IRBEM.Coords()
        alt_lat_lon = _coords_obj.transform(
            X['time'], 
            self.state[f'el{self.sc_id}_pos_gei']/Re, 
            5, 
            0
            )
        
        self.data = {
            'epoch':X['time'],
            'lm':Lm,
            'mlt':MLT,
            'pos_gei':self.state[f'el{self.sc_id}_pos_gei'],
            'lat':alt_lat_lon[:, 1], 
            'lon':alt_lat_lon[:, 2], 
            'alt':alt_lat_lon[:, 0],
        }
        return
    
    def keys(self):
        return self.data.keys()
    
    def _get_kp(self, times):
        """
        Load (and optionally download) the Kp index and resample it to times.
        """
        cdas = cdasws.CdasWs()
        try:
            time_range = cdasws.TimeInterval(
                times[0].replace(tzinfo=timezone.utc), 
                times[-1].replace(tzinfo=timezone.utc)
                )
        except AttributeError as err:
            if "'numpy.datetime64' object has no attribute 'replace'" in str(err):
                times = pd.to_datetime(times).to_pydatetime()
                time_range = cdasws.TimeInterval(
                    times[0].replace(tzinfo=timezone.utc), 
                    times[-1].replace(tzinfo=timezone.utc)
                    )
            else:
                raise
        _, data =  cdas.get_data(
            'OMNI2_H0_MRG1HR', ['KP1800'], time_range
            )
        if isinstance(data['Epoch'][0], datetime):
            pass  # The newest cdasws format.
        elif isinstance(data['KP'].Epoch.data, np.ndarray):
            # An old cdasws format.
            data['Epoch'] = data['KP'].Epoch.data
        else:
            # An old cdasws format.
            data['Epoch'] = cdflib.cdfepoch.to_datetime(data['KP'].Epoch.data)
        kp = pd.DataFrame(index=data['Epoch'], data={'Kp': data['KP']})
        state_times = pd.DataFrame(index=times)
        state_times = pd.merge_asof(
            state_times, kp, left_index=True, right_index=True, 
            tolerance=pd.Timedelta('1h'), direction='backward'
            )
        return state_times.Kp.to_numpy()


class EPD_PAD:
    def __init__(
            self,
            sc_id: str,
            time_range:Tuple[datetime],
            accumulate: float = 0.5,
            pa_bins: np.ndarray = np.arange(0, 181, 20),
            start_pa:int=0,
            spin_time_tol:tuple=(2.5, 3.5),
            min_counts:int=10,
            lc_exclusion_angle:float=10,
            nflux:bool=True,
            t89:bool=False
            ) -> None:
        """
        Calculate the electron pitch angle distribution taken by ELFIN's EPD instrument.

        Parameters
        ----------
        sc_id: str
            The spacecraft id, either "A" or "B". Case insensitive.
        time_range: tuple of datetime or str
            The start and end date and time to load the data.
        accumulate: float
            Number of spin(s) to accumulate over. Currently supports either 0.5 or 1.
        pa_bins: np.ndarray
            The pitch angle bins.
        start_pa: int
            The phase between the time bins and the spin. If start_pa=0, the time bins span 
            0 < PA < 180. If center_pa=90, then the time bins alternatively span 90->0->90 and 
            90->180->90 PAs.
        spin_time_tol: tuple
            The range of valid spin periods, in seconds. The optimal spin period of 3 seconds. 
            If the spin period is outside the tolerance, the corresponding PADS will be marked 
            as NaN.
        min_counts: int
            Fluxes with corresponding counts less than min_counts will be masked as np.nan. If
            None, no masking is applied.
        lc_exclusion_angle: float
            Discard part of the PAD that is within this angle from the loss cone or anti-loss cone angles.
        nflux:bool,
            Load and bin the number flux if True, or energy flux is false.

        Methods
        -------


        Attributes
        ----------
        epd: EPD
            The Level 2 EPD data structure. Has .keys() and .attrs() methods.

        Examples
        --------
        # Plot the ELFIN L2 fluxes.
        import matplotlib.pyplot as plt

        from pad.analysis import _pad

        time_range = ('2020-08-06T13:00', '2020-08-06T13:05')
        sc_id = 'A'
        pad_obj = _pad.EPD_PAD(sc_id, time_range, start_pa=90)
        
        fig, ax = plt.subplots(7, 1, sharex=True, figsize=(7, 9))
        ax[0].set_title(
            f'ELFIN-{sc_id.upper()} | {time_range[0]}-{time_range[1]}'
            f'\nElectron Pitch Angle Distributions'
            )
        pad_obj.plot_omni(ax[0])
        pad_obj.plot_pad_scatter(ax[1])
        pad_obj.plot_pad_spectrogram(ax[2])
        pad_obj.plot_pad_spectrogram(ax[3], energy=520)
        pad_obj.plot_pad_spectrogram(ax[4], energy=1081)
        pad_obj.plot_pad_spectrogram(ax[5], energy=2121)
        pad_obj.plot_blc_dlc_ratio(ax[-1])
        pad_obj.plot_position(ax[-1])
        plt.subplots_adjust(bottom=0.127, right=0.927, top=0.948, hspace=0.133)
        plt.show()

        # Another example to plot the fluxes or counts.
        import matplotlib.pyplot as plt
        from matplotlib.dates import num2date

        import elfinasi

        plot_flux = True
        vmin=1

        time_range = ('2021-11-04T08:12', '2021-11-04T08:20')
        sc_id = 'B'
        # time_range = ('2021-11-04T05:14', '2021-11-04T05:22')
        # sc_id = 'A'
        # time_range = ('2021-07-17T16:32', '2021-07-17T16:38')
        # sc_id = 'B'
        pad_obj = elfinasi.EPD_PAD(
            sc_id, time_range, start_pa=0, min_counts=1, spin_time_tol=(2.5, 3.5), accumulate=1
            )
            
        fig, ax = plt.subplots(7, 1, sharex=True, figsize=(7, 8))
        ax[0].set_title(
            f'ELFIN-{sc_id.upper()} | {time_range[0]}-{time_range[1]}'
            f'\nElectron Pitch Angle Distributions'
            )
        pad_obj.plot_omni(ax[0], flux=plot_flux, vmin=vmin)
        pad_obj.plot_pad_scatter(ax[1], flux=plot_flux, vmin=vmin, energy=63)
        pad_obj.plot_pad_spectrogram(ax[2], flux=plot_flux, vmin=vmin, energy=63)
        pad_obj.plot_pad_spectrogram(ax[3], flux=plot_flux, vmin=vmin, energy=138)
        pad_obj.plot_pad_spectrogram(ax[4], flux=plot_flux, vmin=vmin, energy=305)
        pad_obj.plot_pad_spectrogram(ax[5], flux=plot_flux, vmin=vmin, energy=520)
        pad_obj.plot_blc_dlc_ratio(ax[-1], vmax=1, cmap='viridis')
        pad_obj.plot_position(ax[-1])
        ax[-1].set_xlim(*pad_obj.time_range)

        for ax_i in ax:
            ax_i.fmt_xdata = lambda x: num2date(x).replace(tzinfo=None).isoformat()
        plt.subplots_adjust(bottom=0.168, right=0.927, top=0.948, hspace=0.133)
        plt.show()
        """
        self.sc_id = sc_id.upper()
        self.time_range = validate_time_range(time_range)
        self.accumulate = accumulate
        self.pa_bins = pa_bins
        self.spin_time_tol = spin_time_tol
        self.min_counts = min_counts
        self.lc_exclusion_angle = lc_exclusion_angle
        self.nflux = nflux
        self.t89 = t89
        if self.nflux == True:
            self._flux_var = 'nflux'
        else:
            self._flux_var = 'eflux'

        if isinstance(self.accumulate, (float, int)):
            assert self.accumulate in [0.5, 1], 'Only supports PA accumulation over 0.5 or 1 spins.'
        assert self.sc_id in ['A', 'B'], 'ELFIN spacecraft ID must be either A or B.'
        self.start_pa = int(start_pa)
        assert self.start_pa in [0, 90], (
            f'{start_pa=} is an invalid time binning option. The valid options are: [0, 90].'
            )
        self.lc_type = namedtuple('lc_type', ['time', 'angle'])

        self.epd_l2 = EPD(self.sc_id, self.time_range[0], level=2)
        self.epd_l1 = EPD(self.sc_id, self.time_range[0], level=1)
        self.state = MagEphem(self.sc_id, self.time_range[0], t89=self.t89)    
        self.bin()
        self.precipitation_components()

        self._energy_units = self.epd_l2.attrs(f'el{self.sc_id}_pef_energies_mean')['UNITS']
        self._counts_units = self.epd_l1.attrs(f'el{self.sc_id}_pef')['LABLAXIS']
        self._flux_units = self.epd_l2.attrs(f'el{self.sc_id}_pef_fs_Epat_{self._flux_var}')['UNITS']
        return

    def keys(self):
        return self.epd_l1.keys() + self.epd_l2.keys()

    def bin(self) -> xr.DataArray:
        """
        Bin the electron counts and flux observed by ELFIN-EPD into time and pitch angle bins.

        Returns
        -------
        xr.DataArray
            The PAD array with time, PA, and energy coordinates.
        """
        self.energy = self.epd_l2[f'el{self.sc_id}_pef_energies_mean'].astype(int)
        self.energy_widths =np.stack((
            self.epd_l2[f'el{self.sc_id}_pef_energies_min'], 
            self.epd_l2[f'el{self.sc_id}_pef_energies_max']
            )).T.astype(int)
        self._flux_std_keys = np.array([f'{_energy}_flux_std' for _energy in self.energy])
        self._flux_keys = np.array([f'{_energy}_flux' for _energy in self.energy])
        self._counts_keys = np.array([f'{_energy}_counts' for _energy in self.energy])

        l2_data = {key: self.epd_l2[f'el{self.sc_id}_pef_Et_{self._flux_var}'][:, i]
                for i, key in enumerate(self._flux_keys)}
        l1_data = {f'{energy}_counts': self.epd_l1[f'el{self.sc_id.lower()}_pef'][:, i]
            for i, energy in enumerate(self.energy)}
        l1_len = l1_data[self._counts_keys[0]].shape[0]
        l2_len = l2_data[self._flux_keys[0]].shape[0]

        if l1_len == l2_len:
            # The simple case where the L1 and L2 EPD flux time stamps are identical.
            self.pa_flattened = pd.DataFrame(
                index=self.epd_l2[f'el{self.sc_id}_pef_et_time'], 
                data=l1_data | l2_data  # FYI: "|" combines two dictionaries into one.
                )
            self.pa_flattened['pa'] = self.epd_l2[f'el{self.sc_id}_pef_pa']
        else:
            # Sometimes the L2 data files throw out a few time stamps for some instrumental
            # reason. So here we merge the fluxes together. Example of this is:
            # time_range = ('2021-07-17T16:32', '2021-07-17T16:38')
            # sc_id = 'B'
            l1_df = pd.DataFrame(index=self.epd_l1[f'epoch'], data=l1_data)
            l2_df = pd.DataFrame(index=self.epd_l2[f'el{self.sc_id}_pef_et_time'], data=l2_data)
            l2_df['pa'] = self.epd_l2[f'el{self.sc_id}_pef_pa']

            # Remove time stamps that go back in time. They are not to be trusted.
            dt = (l1_df.index[1:] - l1_df.index[:-1]).total_seconds()
            reverse_time_idt = np.where(dt < 0)[0]
            reverse_time_idt = np.concatenate((reverse_time_idt+1, reverse_time_idt, reverse_time_idt-1))
            l1_df = l1_df.drop(index=l1_df.index[reverse_time_idt+1])
            self.pa_flattened = pd.merge_asof(
                l2_df, l1_df, left_index=True, right_index=True, tolerance=pd.Timedelta(seconds=0.2)
                )
            
            if len(reverse_time_idt) > 0:
                warnings.warn(
                    f'There are {reverse_time_idt.shape[0]} reverse time stamps that were '
                    'dropped from the L1 count data.'
                    )

        for i, (_std_key, _flux_key) in enumerate(zip(self._flux_std_keys, self._flux_keys)):
            self.pa_flattened[_std_key] = self.epd_l2[f'el{self.sc_id}_pef_Et_dfovf'][:, i]*\
                self.pa_flattened[_flux_key]
        self.pa_flattened = self.pa_flattened.loc[
            (self.pa_flattened.index > self.time_range[0]) &
            (self.pa_flattened.index < self.time_range[1])
        ]

        if self.pa_flattened.shape[0] == 0:
            raise ValueError(f'No ELFIN-{self.sc_id} L2 data between {self.time_range[0]} and {self.time_range[1]}.')
        
        if self.min_counts is not None:
            self._mask_low_counts()

        if self.start_pa == 90:
            peak_idt = scipy.signal.find_peaks(
                -np.abs((self.pa_flattened.pa-self.start_pa))
                )[0]
            # Finding 90 degree crossings doubles the accumulate period, so we want to downsample for
            # consistency (it will upsampled again in the "elif self.accumulate == 0.5" block below.)
            peak_idt = peak_idt[::2]  
        elif self.start_pa == 0:
            peak_idt = scipy.signal.find_peaks(self.pa_flattened.pa)[0]
        else:
            raise NotImplementedError

        if self.accumulate == 1:
            pass
        elif self.accumulate == 0.5:
            # Find the intermediate times.
            half_spin_idt = peak_idt[:-1]+((peak_idt[1:]-peak_idt[:-1])/2).astype(int)
            all_idt = np.empty((half_spin_idt.shape[0] + peak_idt.shape[0]), dtype=int)
            all_idt[0::2] = peak_idt
            all_idt[1::2] = half_spin_idt
            peak_idt = all_idt
            self.spin_time_tol = np.array(self.spin_time_tol) / 2
        else:
            raise NotImplementedError
        
        self.time_bins = self.pa_flattened.index[peak_idt]

        _pad_cols = np.concatenate((self._flux_keys, self._flux_std_keys, self._counts_keys))

        empty_data = np.nan*np.zeros(
            (self.time_bins.shape[0]-1, self.pa_bins.shape[0]-1, _pad_cols.shape[0])
            )
        
        self.pad = xr.DataArray(
            empty_data,
            coords=[self.time_bins[:-1], self.pa_bins[:-1], _pad_cols],
            dims=["time", "pa", 'energy']
            )

        for start_time, end_time in zip(self.time_bins[:-1], self.time_bins[1:]):
            # The timedelta makes sure that the filtered DataArray does not include the final data point.
            try:
                time_binned = self.pa_flattened.loc[start_time:end_time-timedelta(seconds=1)]
            except KeyError as err:
                if 'timestamp' in str(err).lower():  # Sometimes the pandas time slicing fails...
                    time_binned = self.pa_flattened.loc[start_time:end_time]
                else:
                    raise
            for _start_pa, _end_pa in zip(self.pa_bins[:-1], self.pa_bins[1:]):
                time_pa_binned = time_binned.loc[
                    (time_binned['pa'] >= _start_pa) & (time_binned['pa'] < _end_pa),
                    ]
                if time_pa_binned.shape[0] == 0:
                    continue
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    self.pad.loc[start_time, _start_pa, self._flux_keys] = \
                        np.nanmean(time_pa_binned[self._flux_keys], axis=0)
                    self.pad.loc[start_time, _start_pa, self._counts_keys] = \
                        np.nansum(time_pa_binned[self._counts_keys], axis=0)
                    self.pad.loc[start_time, _start_pa, self._flux_std_keys] = \
                        np.sqrt(np.sum(np.power(time_pa_binned[self._flux_std_keys], 2), axis=0))
        
        dt = (self.time_bins[1:] - self.time_bins[:-1]).total_seconds()
        invalid_spins = np.where((dt < self.spin_time_tol[0]) | (dt > self.spin_time_tol[1]))[0]
        self.pad[invalid_spins, ...] = np.nan

        if self.pad.shape[0] == invalid_spins.shape[0]:
            warnings.warn(
                f'The ELFIN spin period {dt.to_numpy().mean():.1f} is outside of the '
                f'{self.spin_time_tol} second tolerance. All ELFIN PAD values are NaN.'
                )
        return self.pad
    
    def _mask_low_counts(self):
        """
        Mask the low fluxes and count rates when the corresponding count rate is less than 
        self.min_counts.
        """
        # Need a for loop to avoid indexing hell.
        for _count_key, _flux_key in zip(self._counts_keys, self._flux_keys):
            idx = self.pa_flattened.loc[:, _count_key] < self.min_counts
            self.pa_flattened.loc[idx, _count_key] = np.nan
            self.pa_flattened.loc[idx, _flux_key] = np.nan
        return

    def precipitation_components(self):
        """
        Calculate the bounce loss cone (blc), drift loss cone (dlc), and backscatter (ablc) 
        electron fluxes and standard deviations. Creates 8 attributes: blc, dlc, ablc, blc_std,
        dlc_std, ablc_std, precipitation_ratio, and precipitation_ratio_std
        
        Returns
        -------
        np.array:
            The fluxes inside the BLC.
        np.array:
            The fluxes inside the anti-BLC.
        np.array:
            The fluxes inside the DLC.
        """

        # if hasattr(self, 'blc') and (self._lc_exclusion_angle == lc_exclusion_angle):
        #     # Don't rerun calculation if it has already been done with the same loss 
        #     # cone exclusion angle.
        #     return self.blc, self.ablc, self.dlc
        
        self.blc = np.nan*np.zeros((self.pad.time.shape[0], self.pad.energy.shape[0]//3))
        self.ablc = np.nan*np.zeros((self.pad.time.shape[0], self.pad.energy.shape[0]//3))
        self.dlc = np.nan*np.zeros((self.pad.time.shape[0], self.pad.energy.shape[0]//3))
        self.blc_counts = np.nan*np.zeros((self.pad.time.shape[0], self.pad.energy.shape[0]//3))
        self.ablc_counts = np.nan*np.zeros((self.pad.time.shape[0], self.pad.energy.shape[0]//3))
        self.dlc_counts = np.nan*np.zeros((self.pad.time.shape[0], self.pad.energy.shape[0]//3))
        self.blc_std = np.nan*np.zeros((self.pad.time.shape[0], self.pad.energy.shape[0]//3))
        self.ablc_std = np.nan*np.zeros((self.pad.time.shape[0], self.pad.energy.shape[0]//3))
        self.dlc_std = np.nan*np.zeros((self.pad.time.shape[0], self.pad.energy.shape[0]//3))

        # Southern hemisphere
        ida_southern = np.where(self.lc.angle <= 90)[0]
        zipped = zip(ida_southern, self.lc.angle[ida_southern], self.alc.angle[ida_southern])
        for i, lc_angle, alc_angle in zipped:
            lc_idx = np.where(self.pad.pa <= lc_angle-self.lc_exclusion_angle)[0]
            alc_idx = np.where(self.pad.pa >= alc_angle+self.lc_exclusion_angle)[0]
            dlc_idx = np.where(
                (self.pad.pa >= lc_angle+self.lc_exclusion_angle) &
                (self.pad.pa <= alc_angle-self.lc_exclusion_angle)
                )[0]
            with warnings.catch_warnings(action="ignore"):
                self._calc_blc_ablc_dlc(i, lc_idx, alc_idx, dlc_idx)
                
        # Northern hemisphere
        ida_northern = np.where(self.lc.angle >= 90)[0]
        zipped = zip(ida_northern, self.lc.angle[ida_northern], self.alc.angle[ida_northern])
        for i, lc_angle, alc_angle in zipped:                
            lc_idx = np.where(self.pad.pa >= lc_angle+self.lc_exclusion_angle)[0]
            alc_idx = np.where(self.pad.pa <= alc_angle-self.lc_exclusion_angle)[0]
            dlc_idx = np.where(
                (self.pad.pa <= lc_angle-self.lc_exclusion_angle) &
                (self.pad.pa >= alc_angle+self.lc_exclusion_angle)
                )[0]
            with warnings.catch_warnings(action="ignore"):
                self._calc_blc_ablc_dlc(i, lc_idx, alc_idx, dlc_idx)
        # Ignore the "invalid value encountered in divide" warning
        with np.errstate(divide='ignore', invalid='ignore'):
            self.precipitation_ratio = self.blc/self.dlc
        if np.all(np.isnan(self.precipitation_ratio)):
            warnings.warn(
                'The BLC/DLC ratios are all NaNs. This could be due to the lc_exclusion_angle '
                'excluding all pitch angles sampled.'
                )
        # A reminder on how to propagate uncertainties.
        # https://www.geol.lsu.edu/jlorenzo/geophysics/uncertainties/Uncertaintiespart2.html
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            relative_ratio_std = np.sqrt(
                (self.blc_std/self.blc)**2 + 
                (self.dlc_std/self.dlc)**2
                )
        self.precipitation_ratio_std = relative_ratio_std*self.precipitation_ratio
        return self.blc, self.ablc, self.dlc
    
    def _calc_blc_ablc_dlc(self, idt, lc_idx, alc_idx, dlc_idx):
        self.blc[idt, :] =  np.nanmean(self.pad.isel(time=idt, pa=lc_idx).sel(energy=self._flux_keys), axis=0)
        self.ablc[idt, :] = np.nanmean(self.pad.isel(time=idt, pa=alc_idx).sel(energy=self._flux_keys), axis=0)
        self.dlc[idt, :] =  np.nanmean(self.pad.isel(time=idt, pa=dlc_idx).sel(energy=self._flux_keys), axis=0)

        self.blc_counts[idt, :] =  np.nansum(self.pad.isel(time=idt, pa=lc_idx).sel(energy=self._counts_keys), axis=0)
        self.ablc_counts[idt, :] = np.nansum(self.pad.isel(time=idt, pa=alc_idx).sel(energy=self._counts_keys), axis=0)
        self.dlc_counts[idt, :] =  np.nansum(self.pad.isel(time=idt, pa=dlc_idx).sel(energy=self._counts_keys), axis=0)

        self.blc_std[idt, :] = np.sqrt(np.sum(np.power(
            self.pad.isel(time=idt, pa=lc_idx).sel(energy=self._flux_std_keys), 2
            ), axis=0))
        self.ablc_std[idt, :] = np.sqrt(np.sum(np.power(
            self.pad.isel(time=idt, pa=alc_idx).sel(energy=self._flux_std_keys), 2
            ), axis=0))
        self.dlc_std[idt, :] = np.sqrt(np.sum(np.power(
            self.pad.isel(time=idt, pa=dlc_idx).sel(energy=self._flux_std_keys), 2
            ), axis=0))
        return
        
    def plot_omni(
            self, 
            ax:plt.Axes,
            corrected_flux:np.ndarray=None, 
            flux:bool=True, 
            vmin:float=None, 
            vmax:float=None, 
            labels:bool=True, 
            colorbar:bool=True,
            shrink_cbar=0.9,
            fraction=0.15,
            cax=None,
            pretty_plot:bool=True
            ):
        """
        Plot the spin-averaged electron flux spectrogram.

        Parameters
        ----------
        ax: plt.Axes
            The subplot object to plot on.
        corrected_flux: np.ndarray
            Can use to override the spin-averaged fluxes.
        flux: bool
            Plot the counts or flux.
        vmin: float
            The minimum value to plot of the logarithmic color scale.
        vmax: float
            The maximum value to plot of the logarithmic color scale.
        labels:bool
            Add x- and y-axis labels.
        colorbar: bool
            Add a colorbar.
        """
        if corrected_flux is None:
            if flux:
                z = self.pad.mean(dim='pa', skipna=True).sel(energy=self._flux_keys).T
                label=f"Electron Flux\n{self._flux_units}"
            else:
                z = self.pad.mean(dim='pa', skipna=True).sel(energy=self._counts_keys).T
                label=self._counts_units

            if np.prod(z.shape) != 0:
                p = ax.pcolormesh(
                    self.pad.time, self.energy, z,
                    norm=matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax)
                    )
            else:
                warnings.warn('No valid omnidirectional fluxes.')
                p = None
        else:
            z = corrected_flux
            label = f"Electron Flux\n{self._flux_units}"
            p = ax.pcolormesh(
                z.index, z.columns, z.values.T,
                norm=matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax)
            )

        if labels:
            ax.set(
                yscale='log', 
                ylabel=f"Energy\n{self._energy_units}"
                )
        if colorbar and (not np.all(np.isnan(z))) and (np.prod(z.shape) != 0):
            _cbar = plt.colorbar(p, ax=ax, shrink=shrink_cbar, fraction=fraction)
            _cbar.set_label(label=label, size=6)
        else:
            _cbar = None

        if pretty_plot:
            _text = ax.text(
                0.01, 0.99, f'Spin-averaged', transform=ax.transAxes, va='top'
                )
            _text.set_bbox(dict(facecolor='white', linewidth=0, pad=0.1, edgecolor='k'))
        return p, _cbar
        
    def plot_pad_scatter(
            self, 
            ax:plt.Axes, 
            energy:int=63,
            flux=True,
            lc_lines:bool=True, 
            lc_label_size:int=6,
            size:int=3, 
            vmin:float=None, 
            vmax:float=None,
            pretty_plot:bool=True,
            colorbar:bool=True
        ) -> (matplotlib.collections.PathCollection, matplotlib.colorbar.Colorbar):
        """
        Plot the pitch angle distribution fluxes as a scatter plot.

        Parameters
        ----------
        ax: plt.Axes
            The subplot object to plot on.
        energy: int
            The electron energy in keV.
        flux: bool
            Plot the counts or flux.
        lc_lines:bool  
            Toggles if the loss and anti-loss cone lines
        lc_label_size: int
            The size of the loss and anti-loss cone labels.
        size: int
            The scatter point size, passed directly into plt.scatter().
        vmin: float
            The minimum value to plot of the logarithmic color scale.
        vmax: float
            The maximum value to plot of the logarithmic color scale.
        pretty_plot: bool
            If True, will add annotations and labels.
        
        Returns
        -------
        matplotlib.collections.PathCollection
            The scatter plot object.
        matplotlib.colorbar.Colorbar
            The colorbar object
        """
        if energy is None:
            colors = 'k'
        else:
            _key, _label = self._get_energy_label(flux, energy)
            colors = self.pa_flattened[_key]
        
        scat = ax.scatter(
            self.pa_flattened.index, 
            self.pa_flattened['pa'], 
            c=colors, 
            s=size, 
            norm=matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax)
            )
        if colorbar and (energy is not None) and (not np.all(self.pa_flattened[_key]==0)):
            # Sometimes all flux values are 0. This seems to be during times
            # when that ELFIN-EPD channel was errorous. 
            _cbar = plt.colorbar(scat, ax=ax)
            _cbar.set_label(label=_label, size=6)
        else:
            _cbar = None

        if pretty_plot:
            _text = ax.text(
                0.01, 0.99, f'{energy} keV', transform=ax.transAxes, va='top'
                )
            _text.set_bbox(dict(facecolor='white', linewidth=0, pad=0.1, edgecolor='k'))
            ax.set(ylim=(0, 180), facecolor='grey', ylabel=f'Pitch angle\n[$\\circ$]')
            ax.set_yticks(np.arange(0, 181, 30))

        if lc_lines:
            self._plot_lc_lines(ax, label_size=lc_label_size)
        return scat, _cbar
    
    def plot_pad_spectrogram(
            self, 
            ax:plt.Axes, 
            energy:int=63,
            flux=True,
            lc_lines:bool=True,
            lc_label_size:int=6,
            vmin:float=None, 
            vmax:float=None,
            pretty_plot:bool=True
        ) -> (matplotlib.collections.PathCollection, matplotlib.colorbar.Colorbar):
        """
        Plot the pitch angle distribution spectrogram.
            
        Parameters
        ----------
        ax: plt.Axes
            The subplot object to plot on.
        energy: int
            The electron energy in keV.
        flux: bool
            Plot the counts or flux.
        lc_lines:bool  
            Toggles if the loss and anti-loss cone lines
        lc_label_size: int
            The size of the loss and anti-loss cone labels.
        vmin: float
            The minimum value to plot of the logarithmic color scale.
        vmax: float
            The maximum value to plot of the logarithmic color scale.
        pretty_plot: bool
            If True, will add annotations and labels.
        
        Returns
        -------
        matplotlib.collections.PathCollection
            The scatter plot object.
        matplotlib.colorbar.Colorbar
            The colorbar object
        """
        _key, _label = self._get_energy_label(flux, energy)
        p = ax.pcolormesh(
            self.pad.time,
            self.pad.pa,
            self.pad.sel(energy=_key).T,
            norm=matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax),
            )
        if not np.all(np.isnan(self.pad.sel(energy=_key))): # Color bar will fail if the pad is all nans.
            _cbar = plt.colorbar(p, ax=ax)
            _cbar.set_label(label=_label, size=6)
        else:
            _cbar = None

        if pretty_plot:
            _text = ax.text(
                0.01, 0.99, f'{energy} keV', transform=ax.transAxes, va='top'
                )
            _text.set_bbox(dict(facecolor='white', linewidth=0, pad=0.1, edgecolor='k'))
            ax.set(ylim=(0, 180), facecolor='grey', ylabel=f'Pitch angle\n[$\\circ$]')
            ax.set_yticks(np.arange(0, 181, 30))

        if lc_lines:
            self._plot_lc_lines(ax, label_size=lc_label_size)
        return p, _cbar

    def _plot_lc_lines(self, ax, label_size=6):
        ax.plot(self.lc.time, self.lc.angle, ls='solid', c='w')
        ax.plot(self.alc.time, self.alc.angle, ls='dashed', c='w')
        if self.lc.angle[0] < 90:
            lc_va='top'
            alc_va='bottom'
        else:
            lc_va='bottom'
            alc_va='top'
        ax.text(
            self.lc.time[0], self.lc.angle[0], 'Loss cone', va=lc_va, size=label_size, color='white'
            )
        ax.text(
            self.alc.time[0], self.alc.angle[0], 'Anti-loss cone', va=alc_va, size=label_size, color='white'
            )

    def plot_blc_dlc_ratio(
            self,
            ax:plt.Axes, 
            cmap='viridis',
            vmin:float=1E-1, 
            vmax:float=1,
            labels:bool=True, 
            colorbar:bool=True,
            fraction=0.15,
            shrink_cbar=0.9
            ) -> (matplotlib.collections.PathCollection, matplotlib.colorbar.Colorbar):
        """
        Plot the ratio between the fluxes in the bounce loss cone and the drift loss cone (also
        called "trapped" by the ELFIN team).

        Parameters
        ----------
        ax: plt.Axes
            The subplot object to plot on.
        vmin: float
            The minimum value to plot of the logarithmic color scale.
        vmax: float
            The maximum value to plot of the logarithmic color scale.
        labels:bool
            Add x- and y-axis labels.
        colorbar: bool
            Add a colorbar.
        
        Returns
        -------
        matplotlib.collections.PathCollection
            The scatter plot object.
        matplotlib.colorbar.Colorbar
            The colorbar object
        """
        # Ignore the "invalid value encountered in divide" warning
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = self.blc/self.dlc
        if np.prod(ratio.shape) != 0:
            p =ax.pcolormesh(
                self.pad.time, 
                self.energy, 
                ratio.T,
                shading='nearest',
                norm=matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax),
                cmap=cmap
            )
        else:
            warnings.warn('No valid fluxes.')
            p = None
        if colorbar and np.prod(ratio.shape) != 0:
            _cbar = plt.colorbar(p, ax=ax, shrink=shrink_cbar, fraction=fraction)
            _cbar.set_label(label='BLC/DLC ratio', size=8)
        else:
            _cbar = None
        if labels:
            ax.set(ylabel='Energy\n[keV]', yscale='log')
        return p, _cbar
    
    def plot_blc_dlc_ratio_std(
            self,
            ax:plt.Axes, 
            lc_exclusion_angle:float=10,
            cmap='viridis',
            vmin:float=1E-1, 
            vmax:float=1,
            labels:bool=True, 
            colorbar:bool=True
        ) -> (matplotlib.collections.PathCollection, matplotlib.colorbar.Colorbar):
        """
        Plot the Poisson standard deviation of the ratio between the fluxes in the bounce loss cone
        and the drift loss cone (also sometimes called "trapped" by the ELFIN team).

        Parameters
        ----------
        ax: plt.Axes
            The subplot object to plot on.
        lc_exclusion_angle: float
            Discard part of the PAD that is within this angle from the loss cone or anti-loss cone 
            angles.
        vmin: float
            The minimum value to plot of the logarithmic color scale.
        vmax: float
            The maximum value to plot of the logarithmic color scale.
        labels:bool
            Add x- and y-axis labels.
        colorbar: bool
            Add a colorbar.
        
        Returns
        -------
        matplotlib.collections.PathCollection
            The scatter plot object.
        matplotlib.colorbar.Colorbar
            The colorbar object
        """
        p = ax.pcolormesh(
            self.pad.time, 
            self.energy, 
            self.precipitation_ratio_std.T,
            shading='nearest',
            norm=matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax),
            cmap=cmap
            )
        if colorbar:
            _cbar = plt.colorbar(p, ax=ax)
            _cbar.set_label(label='BLC/DLC ratio error', size=8)
        if labels:
            ax.set(ylabel='Energy\n[keV]', yscale='log', facecolor='grey')
            ax.text(0.01, 0.98, f'$\\sigma_{{BLC/DLC}}$', transform=ax.transAxes, va='top', fontsize=15, color='white')
        return


    def plot_blc(
            self,
            ax:plt.Axes, 
            lc_exclusion_angle:float=10,
            vmin:float=None, 
            vmax:float=None,
            pretty_plot:bool=True
            ) -> (matplotlib.collections.PathCollection, matplotlib.colorbar.Colorbar):
        """
        Plot the binned bounce loss cone fluxes.

        Parameters
        ----------
        ax: plt.Axes
            The subplot object to plot on.
        lc_exclusion_angle: float
            Discard part of the PAD that is within this angle from the loss cone or anti-loss cone 
            angles.
        vmin: float
            The minimum value to plot of the logarithmic color scale.
        vmax: float
            The maximum value to plot of the logarithmic color scale.
        pretty_plot: bool
            If True, will add annotations and labels.
        
        Returns
        -------
        matplotlib.collections.PathCollection
            The scatter plot object.
        matplotlib.colorbar.Colorbar
            The colorbar object
        """
        p = ax.pcolormesh(
            self.pad.time, 
            self.pad.energy, 
            self.blc.T,
            shading='nearest',
            norm=matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax)
            )
        cbar = plt.colorbar(
            p, 
            ax=ax, 
            label=f'BLC Electron Flux\n{self._flux_units}'
            )

        if pretty_plot:
            ax.set(ylabel='Energy\n[keV]', yscale='log', facecolor='grey')
        return p, cbar
    
    def plot_ablc(
            self,
            ax:plt.Axes, 
            lc_exclusion_angle:float=10,
            vmin:float=None, 
            vmax:float=None,
            pretty_plot:bool=True
            ) -> (matplotlib.collections.PathCollection, matplotlib.colorbar.Colorbar):
        """
        Plot the binned anti-bounce loss cone fluxes.

        Parameters
        ----------
        ax: plt.Axes
            The subplot object to plot on.
        lc_exclusion_angle: float
            Discard part of the PAD that is within this angle from the loss cone or anti-loss cone 
            angles.
        vmin: float
            The minimum value to plot of the logarithmic color scale.
        vmax: float
            The maximum value to plot of the logarithmic color scale.
        pretty_plot: bool
            If True, will add annotations and labels.
        
        Returns
        -------
        matplotlib.collections.PathCollection
            The scatter plot object.
        matplotlib.colorbar.Colorbar
            The colorbar object
        """
        p = ax.pcolormesh(
            self.pad.time, 
            self.pad.energy, 
            self.ablc.T,
            shading='nearest',
            norm=matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax)
            )
        cbar = plt.colorbar(
            p, 
            ax=ax, 
            label=f'ABLC Electron Flux\n{self._flux_units}'
            )

        if pretty_plot:
            ax.set(ylabel='Energy\n[keV]', yscale='log', facecolor='grey')
        return p, cbar
    
    def plot_dlc(
            self,
            ax:plt.Axes, 
            lc_exclusion_angle:float=10,
            vmin:float=None, 
            vmax:float=None,
            pretty_plot:bool=True
            ) -> (matplotlib.collections.PathCollection, matplotlib.colorbar.Colorbar):
        """
        Plot the binned drift loss cone fluxes.

        Parameters
        ----------
        ax: plt.Axes
            The subplot object to plot on.
        lc_exclusion_angle: float
            Discard part of the PAD that is within this angle from the loss cone or anti-loss cone 
            angles.
        vmin: float
            The minimum value to plot of the logarithmic color scale.
        vmax: float
            The maximum value to plot of the logarithmic color scale.
        pretty_plot: bool
            If True, will add annotations and labels.
        
        Returns
        -------
        matplotlib.collections.PathCollection
            The scatter plot object.
        matplotlib.colorbar.Colorbar
            The colorbar object
        """
        

        p = ax.pcolormesh(
            self.pad.time, 
            self.pad.energy, 
            self.dlc.T,
            shading='nearest',
            norm=matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax)
            )
        cbar = plt.colorbar(
            p, 
            ax=ax, 
            label=f'DLC Electron Flux\n{self._flux_units}'
            )

        if pretty_plot:
            ax.set(ylabel='Energy\n[keV]', yscale='log', facecolor='grey')
        return p, cbar
    

    def plot_position(self, ax:plt.Axes):
        """
        Add the ELFIN position labels to the x-axis: L-shell, MLT, Alt, Lat, and Lon.

        Parameters
        ----------
        ax: plt.Axes
            The subplot object to plot on.
        """
        transformed_state = self.transform_state()
        data = pd.DataFrame(
            index=self.state['epoch'],
            data={
                'L [T89]':np.abs(self.state['lm']),
                'MLT':self.state['mlt'],
                f'$\\lambda$ [$\\circ$]':transformed_state.mlat,
                'Alt [km]':transformed_state.alt,
                f'Geo Lat [$\\circ$]':transformed_state.lat,
                f'Geo Lon [$\\circ$]':transformed_state.lon
                }
        )
        manylabels.ManyLabels(ax, data)
        return
    
    def transform_state(self):
        """
        Transformes the ELFIN state from GEI to GEO (lat, lon, alt) coordinates and AACGM 
        magnetic latitude.

        Returns
        -------
        pd.DataFrame
            With time, lat, lon, alt, and mlat columns.
        """
        _coords_obj = IRBEM.Coords()
        alt_lat_lon = _coords_obj.transform(
            self.state['epoch'], 
            self.state['pos_gei']/Re, 
            5, 
            0
            )
        aacgm_lats = aacgmv2.convert_latlon_arr(
                alt_lat_lon[:, 1],
                alt_lat_lon[:, 2],
                alt_lat_lon[:, 0],
                self.state['epoch'][0],
                method_code="G2A",
            )[0]
        self.transformed_state = pd.DataFrame(
            data={
                'lat':alt_lat_lon[:, 1], 
                'lon':alt_lat_lon[:, 2], 
                'alt':alt_lat_lon[:, 0],
                'mlat':aacgm_lats
                },
            index=self.state['epoch']
            )
        return self.transformed_state

    @property
    def lc(self):
        """
        Interpolate the loss cone angle to the binned times.
        """
        if hasattr(self, '_lc'):
            return self._lc
        
        original_lc = self.epd_l2[f'el{self.sc_id}_pef_fs_LCdeg']
        original_lc_times = self.epd_l2[f'el{self.sc_id}_pef_fs_time']
        interp_lc = np.interp(date2num(self.pad.time), date2num(original_lc_times), original_lc)
        self._lc = self.lc_type(time=self.pad.time, angle=interp_lc)
        return self._lc
    
    @property
    def alc(self):
        """
        Interpolate the anti-loss cone angle to the binned times.
        """
        if hasattr(self, '_alc'):
            return self._alc
        
        original_alc = self.epd_l2[f'el{self.sc_id}_pef_fs_antiLCdeg']
        original_alc_times = self.epd_l2[f'el{self.sc_id}_pef_fs_time']
        interp_alc = np.interp(date2num(self.pad.time), date2num(original_alc_times), original_alc)
        self._alc = self.lc_type(time=self.pad.time, angle=interp_alc)
        return self._alc
    
    def _get_energy_label(self, flux: bool, energy: int):
        assert not isinstance(energy, str), f"{energy} can't be a string."
        if flux:
            _key = f'{energy}_flux'
            _label = f'Electron Flux\n{self._flux_units}'
        else:
            _key = f'{energy}_counts'
            _label = self._counts_units

        assert _key in self.pa_flattened.columns, (
            f'{energy} keV is an invalid energy channel. Try one of these '
            f'{self.pa_flattened.columns} (the flux/counts are appended in this function).'
        )
        return _key, _label
    

class EPD_PAD_ARTEMYEV:
    def __init__(self, sc_id, time_range, min_counts=None, lc_exclusion_angle:float=10, t89:bool=False):
        """
        Load the EPD ion data that Anton provided.

        Parameters
        ----------
        lc_exclusion_angle: float
            Discard part of the PAD that is within this angle from the loss cone or anti-loss cone angles.

        Examples
        --------
        sc_id = 'a'
        time_range = ('2022-08-10T02:05', '2022-08-10T02:10')
        _epd_pad_artemyev = EPD_PAD_ARTEMYEV(sc_id, time_range)
        _epd_pad = EPD_PAD(sc_id, time_range, start_pa=0, min_counts=None, accumulate=1)
        _epd_pad_artemyev.precipitation_components()

        fig, ax = plt.subplots(4, 1, sharex=True)
        _epd_pad.plot_omni(ax[0], vmin=1E2, vmax=1E7)
        _epd_pad.plot_blc_dlc_ratio(ax[1], lc_exclusion_angle=0)
        _epd_pad_artemyev.plot_omni(ax[2], vmin=1E2, vmax=1E7)
        _epd_pad_artemyev.plot_blc_dlc_ratio(ax[3], lc_exclusion_angle=0)
        _epd_pad.plot_position(ax[-1])
        ax[0].set_title(
            f'ELFIN-{sc_id.upper()} | EPD electrons and ions\n'
            f'{_epd_pad.time_range[0]:%Y-%m-%d}'
            )
        plt.show()
        """
        self.lc_type = namedtuple('lc_type', ['time', 'angle'])
        self.sc_id = sc_id.lower()
        self.time_range = validate_time_range(time_range)
        self.min_counts = min_counts
        self.lc_exclusion_angle = lc_exclusion_angle
        self.t89 = t89
                
        file_dir = pathlib.Path(elfinasi.data_dir)
        
        cps_filename_glob = f'{self.time_range[0]:%Y-%m-%d}_mike*3dspec_{self.sc_id}_cps*.dat'
        flux_filename_glob = f'{self.time_range[0]:%Y-%m-%d}_mike*3dspec_{self.sc_id}_nflux*.dat'
        cps_file_paths = list(file_dir.rglob(cps_filename_glob))
        flux_file_paths = list(file_dir.rglob(flux_filename_glob))
        assert len(cps_file_paths) == 1 and len(flux_file_paths) == 1, (
            f'{len(cps_file_paths)} count files found and {len(flux_file_paths)} '
            f'flux files found starting in the {file_dir} directory that match '
            f'the {cps_filename_glob} or {flux_filename_glob} glob patterns.'
            )
        cps_file_path = cps_file_paths[0]
        flux_file_path = flux_file_paths[0]
        self.counts = self._load_artemyev_file(cps_file_path)
        self.flux = self._load_artemyev_file(flux_file_path)
        self.flux['flux'] = self.flux.pop('counts')  # Rename key to be consistant.
        self.epd_l2 = EPD(self.sc_id, self.time_range[0], level=2)
        self.state = MagEphem(self.sc_id, self.time_range[0], t89=self.t89)

        self.energy = self.flux['energy']
        self._energy_units = 'keV'
        self._flux_units = '1/cm2/s/sr/MeV'
        self._counts_units = 'Counts/second'

        if self.min_counts is not None:
            self._mask_low_counts()
        self.precipitation_components()
        return

    def _load_artemyev_file(self, path):
        num_lines = sum(1 for _ in open(path))
        count_times = []
        pitch_angles = []
        counts = np.nan*np.zeros((num_lines, 10, 16))  # Bigger than we need.

        if str(self.time_range[0].date()) == '2022-08-19':
            block_size=10  # Xiaofei Shi's data
        else:
            block_size=12

        with open(path, 'r') as f:
            energy = next(f).strip().split(' ')
            energy = np.array(energy).astype(float)
            
            for i, row in enumerate(f):
                if i % block_size == 0:
                    count_times.append(dateutil.parser.parse(row))
                    j = 0  # The counter for the number of spins.
                elif i % block_size == 1:
                    pitch_angles.append(np.array(row.split()).astype(float))
                else:
                    # Accumulate counts in each PA bin.
                    counts[len(count_times), j, :] = np.array(row.split()).astype(float)
                    j += 1
        count_times = np.array(count_times)
        pitch_angles = np.array(pitch_angles)[:, ::-1]  # For some reason the pitch angles are flipped.
        counts = counts[:count_times.shape[0], ...]
        idt = np.where((count_times >= self.time_range[0]) & (count_times <= self.time_range[1]))[0]
        data = {
            'time':count_times[idt],
            'pa':pitch_angles[idt, ...],
            'energy':energy,
            'counts': counts[idt, ...]
        }
        return data
    
    def _mask_low_counts(self):
        """
        Mask the low fluxes and count rates when the corresponding count rate is less than 
        self.min_counts.
        """
        idx = np.where(self.counts['counts'] <= self.min_counts)
        self.counts['counts'][idx] = np.nan
        self.flux['flux'][idx] = np.nan
        return
    
    def precipitation_components(self):
        """
        Calculate the bounce loss cone (blc), drift loss cone (dlc), and backscatter (ablc) 
        electron fluxes and standard deviations. Creates 8 attributes: blc, dlc, ablc, blc_std,
        dlc_std, ablc_std, precipitation_ratio, and precipitation_ratio_std
        
        Returns
        -------
        np.array:
            The fluxes inside the BLC.
        np.array:
            The fluxes inside the anti-BLC.
        np.array:
            The fluxes inside the DLC.
        """
        self.blc = np.nan*np.zeros((self.flux['time'].shape[0], self.flux['energy'].shape[0]))
        self.ablc = np.nan*np.zeros((self.flux['time'].shape[0], self.flux['energy'].shape[0]))
        self.dlc = np.nan*np.zeros((self.flux['time'].shape[0], self.flux['energy'].shape[0]))
        self.blc_counts = np.nan*np.zeros((self.flux['time'].shape[0], self.flux['energy'].shape[0]))
        self.ablc_counts = np.nan*np.zeros((self.flux['time'].shape[0], self.flux['energy'].shape[0]))
        self.dlc_counts = np.nan*np.zeros((self.flux['time'].shape[0], self.flux['energy'].shape[0]))
        self.blc_std = np.nan*np.zeros((self.flux['time'].shape[0], self.flux['energy'].shape[0]))
        self.ablc_std = np.nan*np.zeros((self.flux['time'].shape[0], self.flux['energy'].shape[0]))
        self.dlc_std = np.nan*np.zeros((self.flux['time'].shape[0], self.flux['energy'].shape[0]))

        # Northern hemisphere
        ida_northern = np.where(self.lc.angle <= 90)[0]
        zipped = zip(ida_northern, self.lc.angle[ida_northern], self.alc.angle[ida_northern])
        for i, lc_angle, alc_angle in zipped:
            lc_idx = np.where(self.flux['pa'][i, :] <= lc_angle-self.lc_exclusion_angle)[0]
            alc_idx = np.where(self.flux['pa'][i, :] >= alc_angle+self.lc_exclusion_angle)[0]
            dlc_idx = np.where(
                (self.flux['pa'][i, :] >= lc_angle+self.lc_exclusion_angle) &
                (self.flux['pa'][i, :] <= alc_angle-self.lc_exclusion_angle)
                )[0]
            with warnings.catch_warnings(action="ignore"):
                self._calc_blc_ablc_dlc(i, lc_idx, alc_idx, dlc_idx)
                
        # Southern hemisphere
        ida_southern = np.where(self.lc.angle >= 90)[0]
        zipped = zip(ida_southern, self.lc.angle[ida_southern], self.alc.angle[ida_southern])
        for i, lc_angle, alc_angle in zipped:                
            lc_idx = np.where(self.flux['pa'][i, :] >= lc_angle+self.lc_exclusion_angle)[0]
            alc_idx = np.where(self.flux['pa'][i, :] <= alc_angle-self.lc_exclusion_angle)[0]
            dlc_idx = np.where(
                (self.flux['pa'][i, :] <= lc_angle-self.lc_exclusion_angle) &
                (self.flux['pa'][i, :] >= alc_angle+self.lc_exclusion_angle)
                )[0]
            with warnings.catch_warnings(action="ignore"):
                self._calc_blc_ablc_dlc(i, lc_idx, alc_idx, dlc_idx)
        # Ignore the "invalid value encountered in divide" warning
        with np.errstate(divide='ignore', invalid='ignore'):
            self.precipitation_ratio = self.blc/self.dlc
        relative_ratio_std = np.sqrt(
            (self.blc_std/self.blc)**2 + 
            (self.dlc_std/self.dlc)**2
            )
        self.precipitation_ratio_std = relative_ratio_std*self.precipitation_ratio

        if np.all(np.isnan(self.precipitation_ratio)):
            warnings.warn(
                'The BLC/DLC ratios are all NaNs. This could be due to the lc_exclusion_angle '
                'excluding all pitch angles sampled.'
                )
        # A reminder on how to propagate uncertainties.
        # https://www.geol.lsu.edu/jlorenzo/geophysics/uncertainties/Uncertaintiespart2.html
        # relative_ratio_std = np.sqrt(
        #     (self.blc_std/self.blc)**2 + 
        #     (self.dlc_std/self.dlc)**2
        #     )
        # self.precipitation_ratio_std = relative_ratio_std*self.precipitation_ratio
        return self.blc, self.ablc, self.dlc
    
    def transform_state(self):
        """
        Transformes the ELFIN state from GEI to GEO (lat, lon, alt) coordinates and AACGM 
        magnetic latitude.

        Returns
        -------
        pd.DataFrame
            With time, lat, lon, alt, and mlat columns.
        """
        _coords_obj = IRBEM.Coords()
        alt_lat_lon = _coords_obj.transform(
            self.state['epoch'], 
            self.state['pos_gei']/Re, 
            5, 
            0
            )
        aacgm_lats = aacgmv2.convert_latlon_arr(
                alt_lat_lon[:, 1],
                alt_lat_lon[:, 2],
                alt_lat_lon[:, 0],
                self.state['epoch'][0],
                method_code="G2A",
            )[0]
        self.transformed_state = pd.DataFrame(
            data={
                'lat':alt_lat_lon[:, 1], 
                'lon':alt_lat_lon[:, 2], 
                'alt':alt_lat_lon[:, 0],
                'mlat':aacgm_lats
                },
            index=self.state['epoch']
            )
        return self.transformed_state
    
    def plot_omni(
            self, 
            ax:plt.Axes, 
            flux:bool=True, 
            vmin:float=None, 
            vmax:float=None, 
            labels:bool=True, 
            colorbar:bool=True,
            shrink_cbar=0.9,
            pretty_plot:bool=True
            ):
        """
        Plot the spin-averaged electron flux spectrogram.

        Parameters
        ----------
        ax: plt.Axes
            The subplot object to plot on.
        flux: bool
            Plot the counts or flux.
        vmin: float
            The minimum value to plot of the logarithmic color scale.
        vmax: float
            The maximum value to plot of the logarithmic color scale.
        labels:bool
            Add x- and y-axis labels.
        colorbar: bool
            Add a colorbar.
        """
        if flux:
            z = np.nanmean(self.flux['flux'], axis=1).T
            label=f"Ion Flux\n#/(cm^2*s*str*MeV)"
        else:
            z = np.nanmean(self.counts['counts'], axis=1).T
            label=f"Ion Counts"
        p = ax.pcolormesh(
            self.counts['time'], self.counts['energy'], z,
            norm=matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax)
            )
        if labels:
            ax.set(
                yscale='log', 
                ylabel=f"Energy\n[keV]"
                )
        if colorbar and (not np.all(np.isnan(z))):
            _cbar = plt.colorbar(p, ax=ax, shrink=shrink_cbar)
            _cbar.set_label(label=label, size=6)

        if pretty_plot:
            _text = ax.text(
                0.01, 0.99, f'Spin-averaged', transform=ax.transAxes, va='top'
                )
            _text.set_bbox(dict(facecolor='white', linewidth=0, pad=0.1, edgecolor='k'))
        return p, _cbar
    
    def plot_blc_dlc_ratio(
            self,
            ax:plt.Axes, 
            cmap='viridis',
            vmin:float=1E-1, 
            vmax:float=1,
            labels:bool=True, 
            colorbar:bool=True,
            shrink_cbar=0.9
            ) -> (matplotlib.collections.PathCollection, matplotlib.colorbar.Colorbar):
        """
        Plot the ratio between the fluxes in the bounce loss cone and the drift loss cone (also
        called "trapped" by the ELFIN team).

        Parameters
        ----------
        ax: plt.Axes
            The subplot object to plot on.
        vmin: float
            The minimum value to plot of the logarithmic color scale.
        vmax: float
            The maximum value to plot of the logarithmic color scale.
        labels:bool
            Add x- and y-axis labels.
        colorbar: bool
            Add a colorbar.
        
        Returns
        -------
        matplotlib.collections.PathCollection
            The scatter plot object.
        matplotlib.colorbar.Colorbar
            The colorbar object
        """
        # Ignore the "invalid value encountered in divide" warning
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = self.blc/self.dlc
        p = ax.pcolormesh(
            self.counts['time'], 
            self.counts['energy'], 
            ratio.T,
            shading='nearest',
            norm=matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax),
            cmap=cmap
            )
        if colorbar:
            _cbar = plt.colorbar(p, ax=ax, shrink=shrink_cbar)
            _cbar.set_label(label='BLC/DLC ratio', size=8)
        else:
            _cbar = None
        if labels:
            ax.set(ylabel='Energy\n[keV]', yscale='log', facecolor='grey')
        return p, _cbar
            
    def plot_blc_dlc_ratio_std(
            self,
            ax:plt.Axes, 
            cmap='viridis',
            vmin:float=1E-1, 
            vmax:float=1,
            labels:bool=True, 
            colorbar:bool=True
        ) -> (matplotlib.collections.PathCollection, matplotlib.colorbar.Colorbar):
        """
        Plot the Poisson standard deviation of the ratio between the fluxes in the bounce loss cone
        and the drift loss cone (also sometimes called "trapped" by the ELFIN team).

        Parameters
        ----------
        ax: plt.Axes
            The subplot object to plot on.
        vmin: float
            The minimum value to plot of the logarithmic color scale.
        vmax: float
            The maximum value to plot of the logarithmic color scale.
        labels:bool
            Add x- and y-axis labels.
        colorbar: bool
            Add a colorbar.
        
        Returns
        -------
        matplotlib.collections.PathCollection
            The scatter plot object.
        matplotlib.colorbar.Colorbar
            The colorbar object
        """
        p = ax.pcolormesh(
            self.flux['time'], 
            self.energy, 
            self.precipitation_ratio_std.T,
            shading='nearest',
            norm=matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax),
            cmap=cmap
            )
        if colorbar:
            _cbar = plt.colorbar(p, ax=ax)
            _cbar.set_label(label='BLC/DLC ratio error', size=8)
        if labels:
            ax.set(ylabel='Energy\n[keV]', yscale='log', facecolor='grey')
            ax.text(0.01, 0.98, f'$\\sigma_{{BLC/DLC}}$', transform=ax.transAxes, va='top', fontsize=15, color='white')
        return
    
    def plot_position(self, ax:plt.Axes):
        """
        Add the ELFIN position labels to the x-axis: L-shell, MLT, Alt, Lat, and Lon.

        Parameters
        ----------
        ax: plt.Axes
            The subplot object to plot on.
        """
        transformed_state = self.transform_state()
        data = pd.DataFrame(
            index=self.state['epoch'],
            data={
                'L [T89]':np.abs(self.state['lm']),
                'MLT':self.state['mlt'],
                f'$\\lambda$ [$\\circ$]':transformed_state.mlat,
                'Alt [km]':transformed_state.alt,
                f'Geo Lat [$\\circ$]':transformed_state.lat,
                f'Geo Lon [$\\circ$]':transformed_state.lon
                }
        )
        manylabels.ManyLabels(ax, data)
        return
    
    @property
    def lc(self):
        """
        Interpolate the loss cone angle to the binned times.
        """
        if hasattr(self, '_lc'):
            return self._lc
        
        original_lc = self.epd_l2[f'el{self.sc_id}_pef_fs_LCdeg']
        original_lc_times = self.epd_l2[f'el{self.sc_id}_pef_fs_time']
        interp_lc = np.interp(date2num(self.counts['time']), date2num(original_lc_times), original_lc)
        self._lc = self.lc_type(time=self.counts['time'], angle=interp_lc)
        return self._lc
    
    @property
    def alc(self):
        """
        Interpolate the anti-loss cone angle to the binned times.
        """
        if hasattr(self, '_alc'):
            return self._alc
        
        original_alc = self.epd_l2[f'el{self.sc_id}_pef_fs_antiLCdeg']
        original_alc_times = self.epd_l2[f'el{self.sc_id}_pef_fs_time']
        interp_alc = np.interp(date2num(self.counts['time']), date2num(original_alc_times), original_alc)
        self._alc = self.lc_type(time=self.counts['time'], angle=interp_alc)
        return self._alc

    def _calc_blc_ablc_dlc(self, idt, lc_idx, alc_idx, dlc_idx):
        self.blc[idt, :] =  np.nanmean(self.flux['flux'][idt, lc_idx, :], axis=0)
        self.ablc[idt, :] = np.nanmean(self.flux['flux'][idt, alc_idx, :], axis=0)
        self.dlc[idt, :] =  np.nanmean(self.flux['flux'][idt, dlc_idx, :], axis=0)

        self.blc_counts[idt, :] =  np.nanmean(self.counts['counts'][idt, lc_idx, :], axis=0)
        self.ablc_counts[idt, :] = np.nanmean(self.counts['counts'][idt, alc_idx, :], axis=0)
        self.dlc_counts[idt, :] =  np.nanmean(self.counts['counts'][idt, dlc_idx, :], axis=0)

        self.blc_std[idt, :] = self.blc[idt, :]*np.sqrt(np.nansum(self.counts['counts'][idt, lc_idx, :]))/\
            np.nanmean(self.counts['counts'][idt, lc_idx, :])
        self.ablc_std[idt, :] = self.ablc[idt, :]*np.sqrt(np.nansum(self.counts['counts'][idt, alc_idx, :]))/\
            np.nanmean(self.counts['counts'][idt, alc_idx, :])
        self.dlc_std[idt, :] = self.dlc[idt, :]*np.sqrt(np.nansum(self.counts['counts'][idt, dlc_idx, :]))/\
            np.nanmean(self.counts['counts'][idt, dlc_idx, :])
        return


def map_elfin(df, alt=110, hemi_flag=0):
    """
    Map ELFIN's location along the magnetic field line to alt using IRBEM.MagFields.find_foot_print.

    Parameters
    ----------
    alt: float
        The mapping altitude in units of kilometers
    hemi_flag: int
        What direction to trace the field line: 
        0 = same magnetic hemisphere as starting point
        +1   = northern magnetic hemisphere
        -1   = southern magnetic hemisphere
        +2   = opposite magnetic hemisphere as starting point
    """
    m = IRBEM.MagFields(kext='T89')
    _all = np.zeros_like(df.loc[:, ['alt', 'lat', 'lon']])

    for i, (time, row) in enumerate(df.iterrows()):
        X = {'Time':time, 'x1':row['alt'], 'x2':row['lat'], 'x3':row['lon']}
        _all[i, :] = m.find_foot_point(X, {'Kp':56}, alt, hemi_flag)['XFOOT']
    _all[_all == -1E31] = np.nan
    mapped_df = df.copy()
    mapped_df.loc[:, ['alt', 'lat', 'lon']] = _all
    return mapped_df

if __name__ == '__main__':
    import string

    sc_id = 'b'
    time_range = (datetime(2021, 7, 17, 19, 29, 0), datetime(2021, 7, 17, 19, 45, 0))
    # _epd_pad_artemyev = EPD_PAD_ARTEMYEV(sc_id, time_range, min_counts=10, lc_exclusion_angle=0)
    _epd_pad = EPD_PAD(sc_id, time_range, start_pa=0, min_counts=0, accumulate=1, lc_exclusion_angle=0)

    for flux_key, count_key in zip(_epd_pad._flux_keys, _epd_pad._counts_keys): 
        df = (_epd_pad.pa_flattened[flux_key]/_epd_pad.pa_flattened[count_key])
        plt.hist(
            df.to_numpy().flatten(), 
            bins=np.logspace(0, 6), 
            label=f'{flux_key.split("_")[0]} keV',
            histtype='step'
            )

    plt.legend(loc=1)
    plt.ylabel('Number of samples')
    plt.xlabel('flux/counts')
    plt.title(f'{_epd_pad.time_range[0].date()} | ELFIN-{sc_id.upper()} flux/count ratios')
    plt.yscale('log')
    plt.xscale('log')
    plt.show()

    # fig, ax = plt.subplots(4, 1, sharex=True, figsize=(7, 8))
    # # _epd_pad_artemyev.plot_omni(ax[0], vmin=1E2, vmax=1E7, pretty_plot=False)
    # _epd_pad.plot_omni(ax[1], vmin=1E2, vmax=1E7, pretty_plot=False)
    
    # # _epd_pad_artemyev.plot_blc_dlc_ratio(ax[2], lc_exclusion_angle=0, vmin=1E-2, vmax=1)
    # _epd_pad.plot_blc_dlc_ratio(ax[3], vmin=1E-2, vmax=1)
    # _epd_pad.plot_position(ax[-1])
    # ax[0].set_title(
    #     f'ELFIN-{sc_id.upper()} | EPD electrons and ions | {_epd_pad.time_range[0]:%Y-%m-%d}'
    #     )
    # plt.subplots_adjust(bottom=0.168, right=0.927, top=0.948, hspace=0.133)

    # labels = [
    #     'Omnidirectional ions',
    #     'Omnidirectional electrons',
    #     'Ion precipitation ratio',
    #     'Electron precipitation ratio'
    #     ]
    # for (ax_i, panel_letter, label) in zip(ax, string.ascii_lowercase, labels):
    #     _text = ax_i.text(
    #         0.01, 0.99, f'({panel_letter}) {label}', va='top', transform=ax_i.transAxes, 
    #         fontsize=12
    #         )
    #     _text.set_bbox(dict(facecolor='white', linewidth=0, pad=0.1, edgecolor='k'))
    # plt.show()
    # pass