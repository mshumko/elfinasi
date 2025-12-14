import asilib
import asilib.map
import asilib.asi
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import aacgmv2
import numpy as np

time_range = ('2022-09-04T04:10:00', '2022-09-04T04:30:00')
location_codes = ('ATHA', 'PINA', 'GILL', 'KAPU', 'TPAS')  # 'RANK'

ax = asilib.map.create_map(lon_bounds=(-125, -75), lat_bounds=(40, 63))

asis = asilib.Imagers(
    [asilib.asi.themis(location_code=location_code, time_range=time_range) 
    for location_code in location_codes]
    )
gen = asis.animate_map_gen(overwrite=True, ax=ax)
for _guide_time, _asi_times, _asi_images, _ in gen:
    if 'midnight_line' in locals():
        # midnight_line[0].remove()
        _plot_time.remove()
    # midnight_lon = aacgmv2.convert_mlt(24, _guide_time, m2a=True)[0]
    # fractional_hour = _guide_time.hour + _guide_time.minute/60 + _guide_time.second/3600
    # midnight_lon = (15*(fractional_hour + 12)+180) % 360 - 180
    # midnight_line = ax.plot(
    #     midnight_lon*np.ones(10), np.linspace(0, 90, num=10), transform=ccrs.PlateCarree(), 
    #     c='purple'
    #     )
    _plot_time = ax.text(
        0, 0.99, f'{_guide_time:%H:%M:%S}', va='top', transform=ax.transAxes, fontsize=15
        )