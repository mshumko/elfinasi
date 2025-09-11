"""
Download and load NOAA POES and ESA MetOp data.
"""
import pathlib
import urllib
from datetime import datetime
from typing import List, Tuple

import pandas as pd
import netCDF4

import asilib
import asilib.asi


poes_url = "https://www.ncei.noaa.gov/data/poes-metop-space-environment-monitor/access/"
download_dir = asilib.ASI_DATA_DIR / 'poes'

if not download_dir.exists():
	download_dir.mkdir(parents=True)
	print('Created directory for GPS data:', download_dir)


def download_poes(
        date: datetime, 
        sc_ids: str | List[str]=None, 
        level:str='l1b', 
        version='v01r00', 
        redownload:bool=False
        ) -> Tuple[List[pathlib.Path], List[str]]:
    """
    Download POES & MetOp MEPED data on a given date.

    Parameters
    ----------
    date: datetime
        The date for which to download the weekly GPS data.
    sc_ids: str or list of str (optional)
        The spacecraft ID(s) to download data. If None, it will download data for all available 
        spacecraft IDs. Valid example inputs include: "noaa18", or ["noaa18", "metop01"].
        If None, will load ['noaa18', 'noaa19', 'metop01', 'metop02', 'metop03']
    level: str
        The data level.
    version: str (optional)
        The data version and revision number.
    redownload: bool (optional)
        If True, will redownload existing files.

    Returns
    -------
    file_paths: list of pathlib.Path
        The paths to the downloaded files.
    spacecraft_ids: list of str
        The spacecraft IDs corresponding to the downloaded files.

    Example
    -------
    
    """
    if sc_ids is None:
        sc_ids = ['noaa18', 'noaa19', 'metop01', 'metop02', 'metop03']
    else:
        sc_ids = [sc_ids] if isinstance(sc_ids, str) else sc_ids  # Ensure sc_id is a list.

    if not isinstance(sc_ids, list):
        raise TypeError(f"sc_id must be a list or a string, got {type(sc_ids)}.")
    
    file_paths = []
    spacecraft_ids = []

    for sc_id in sc_ids:
        file_name = f"poes_{sc_id[0]}{sc_id[-2:]}_{date:%Y%m%d}_proc.nc"
        if pathlib.Path(download_dir / file_name).exists() and not redownload:
            file_paths.append(download_dir / file_name)
            spacecraft_ids.append(sc_id)
            continue

        # https://www.ncei.noaa.gov/instruments/solar-space-observing/particle-detectors/sem/poes/access/l1b/v01r00/2021/
        # https://www.ncei.noaa.gov/data/poes-metop-space-environment-monitor/access/l1b/v01r00/2020/noaa18/
        _url = urllib.parse.urljoin(
            poes_url, 
            f"{level}/{version}/{date.year}/{sc_id}/"+file_name, 
            allow_fragments=True
            )

        downloader = asilib.download.Downloader(_url, download_dir=download_dir)
        download_path = downloader.download(redownload=redownload)
        
        file_paths.append(download_path)
        spacecraft_ids.append(sc_id)
    return file_paths, spacecraft_ids

def poes(
        date: datetime, 
        sc_ids: str | List[str]=None, 
        level:str='l1b', 
        version='v01r00', 
        redownload:bool=False
        ) -> dict:
    """
    Load POES & MetOp MEPED data on a given date.

    Parameters
    ----------
    date: datetime
        The date for which to download the weekly GPS data.
    sc_ids: str or list of str (optional)
        The spacecraft ID(s) to download data. If None, it will download data for all available 
        spacecraft IDs. Valid example inputs include: "noaa18", or ["noaa18", "metop01"].
        If None, will load ['noaa18', 'noaa19', 'metop01', 'metop02', 'metop03']
    level: str
        The data level.
    version: str (optional)
        The data version and revision number.
    redownload: bool (optional)
        If True, will redownload existing files.

    Returns
    -------
    dict
        With spacecraft-id (i.e., noaa18, metop02) keys with the corresponding meped data.

    Example
    -------
    
    """
    _paths, _sc_ids = download_poes(
        date, sc_ids=sc_ids, level=level, version=version, redownload=redownload
        )
    data = {}
    
    for _path, _sc_id in zip(_paths, _sc_ids):
        _nc_obj = netCDF4.Dataset(_path)
        data[_sc_id] = attrs_dict()
        for var in _nc_obj.variables:
            data[_sc_id][var] = _nc_obj.variables[var][...]
            data[_sc_id].attrs[var] = _nc_obj.variables[var]
        
        _date_component = [
            f'{_year}-{_doy}' for _year, _doy in 
            zip(_nc_obj.variables['year'][:], _nc_obj.variables['day'][:])
            ]
        _date_objs = pd.to_datetime(_date_component, format='%Y-%j')
        _time_objs = pd.to_timedelta(_nc_obj.variables['msec'][:].data, unit='milliseconds')
        data[_sc_id]['time'] = _date_objs + _time_objs
    return data

class attrs_dict(dict):
    """
    Expand Python's dict class to include an attr attribute dictionary.
    
    Code credit goes to Matt Anderson: https://stackoverflow.com/a/2390997 (blame him for 
    your problems).
    """
    def __init__(self, *args, **kwargs):
        self.update(*args, **kwargs)
        self.attrs = {}
        return


if __name__ == '__main__':
    data = poes(datetime(2021, 11, 4), 'noaa19')
    pass