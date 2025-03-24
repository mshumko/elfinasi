import pathlib

import elfinasi.elfin as elfin
from elfinasi.elfin import State, EPD, MagEphem, EPD_PAD, EPD_PAD_ARTEMYEV

# Create a data and plot directories the first time it runs.
data_dir = pathlib.Path(elfin.__file__).parents[2] / 'data'
if not data_dir.exists():
    data_dir.mkdir()
    print(f'Crated the {data_dir} directory.')

plot_dir = pathlib.Path(elfin.__file__).parents[2] / 'plots'
if not plot_dir.exists():
    plot_dir.mkdir()
    print(f'Crated the {plot_dir} directory.')