"""
A minimal debug module to verify ELFIN's State epoch values using pyspedas/cdflib and spacepy.
"""
import pathlib
import dateutil.parser

import cdflib
import pyspedas
import pyspedas.projects.elfin.config as config  # To find the downloaded CDF file.
import spacepy
import spacepy.pycdf

print(
    f'cdflib version: {cdflib.__version__}, '
    f'pyspedas version: {pyspedas.version()}, '
    f'spacepy version: {spacepy.__version__}'
    )

# elfin_probe = 'a'
# time_range_str = ('2022-09-04T04:15', '2022-09-04T04:25')
elfin_probe = 'b'
time_range_str = ('2022-03-05T15:29', '2022-03-05T15:34')

time_range = (dateutil.parser.parse(time_range_str[0]), dateutil.parser.parse(time_range_str[1]))

pyspedas_elfin_vars = pyspedas.projects.elfin.state(time_range_str, probe=elfin_probe.lower())

file_pattern = f'el{elfin_probe}_l1_state_defn_{time_range[0]:%Y%m%d}_v*.cdf'
file_paths = list(pathlib.Path(config.CONFIG['local_data_dir']).rglob(file_pattern))
assert len(file_paths) == 1, (
    f'Expected to find 1 file matching {file_pattern}, found {len(file_paths)}: {file_paths}.'
    )

cdf_file_path = str(file_paths[0])
pycdf_file = spacepy.pycdf.CDF(cdf_file_path)

print(pyspedas_elfin_vars)
print(pycdf_file.keys())

print(pyspedas.data_quants[f'el{elfin_probe}_pos_gei'].time)
print(pycdf_file[f'el{elfin_probe}_state_time'][:])