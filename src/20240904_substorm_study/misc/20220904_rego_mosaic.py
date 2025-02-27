import asilib
import asilib.map
import asilib.asi
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import aacgmv2
import numpy as np

time_range = ('2022-09-04T04:00:00', '2022-09-04T05:00:00')
location_codes = ('GILL', 'FSIM', 'LUCK', 'RANK', 'TALO')
alt=230

fig = plt.figure(figsize=(6, 7))
ax = asilib.map.create_map(lon_bounds=(-120, -85), lat_bounds=(44, 70), fig_ax=(fig, 111))
plt.subplots_adjust(top=0.95)
plt.tight_layout()

asis = asilib.Imagers(
    [asilib.asi.rego(location_code=location_code, time_range=time_range, alt=alt) 
    for location_code in location_codes]
    )
for _imager in asis.imagers:
    print(_imager.meta['location'], _imager.get_color_bounds())
ax.set_title(f'REGO | {time_range[0][:10]} | {alt} km map altitude')
gen = asis.animate_map_gen(overwrite=True, ax=ax, overlap=False, min_elevation=10)
for _guide_time, _asi_times, _asi_images, _ in gen:
    if '_plot_time' in locals():
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
        0.05, 0.95, f'{_guide_time:%H:%M:%S}', va='top', transform=ax.transAxes, fontsize=15
        )