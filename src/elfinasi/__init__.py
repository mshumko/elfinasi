import pathlib

import elfinasi.elfin as elfin
from elfinasi.elfin import State, EPD, MagEphem, EPD_PAD, EPD_PAD_ARTEMYEV

# Create a data directory the first time it runs.
data_dir = pathlib.Path(elfin.__file__).parents[2] / 'data'
if not data_dir.exists():
    data_dir.mkdir()
    print(f'Crated the {data_dir} directory.')