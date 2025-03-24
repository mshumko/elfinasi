import string

import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import pandas as pd

import elfinasi
from elfinasi import EPD_PAD, EPD_PAD_ARTEMYEV

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.style.use('dark_background')

# time_range = ('2022-08-17T23:43:00','2022-08-17T23:47:00')
# sc_id = 'b'
# time_range = ('2022-09-04T04:17:00','2022-09-04T04:24:00')
# sc_id = 'a'
time_range = ('2022-09-04T03:35:00','2022-09-04T03:40:00')
sc_id = 'b'

    
# try:
_epd_ipad = EPD_PAD_ARTEMYEV(sc_id, time_range, min_counts=5, lc_exclusion_angle=0)
plot_ions = True
n_subplots=4
title_snippet = 'EPD electrons and ions'
labels = [
    'Omnidirectional electrons',
    'Electron precipitation ratio',
    'Omnidirectional ions',
    'Ion precipitation ratio',
    ]
# except Exception as err:  # Bad idea but whatever, we need the plots ASAP!
#     plot_ions = False
#     n_subplots=2
#     title_snippet = 'EPD electrons'
#     labels = [
#         'Omnidirectional electrons',
#         'Electron precipitation ratio',
#         ]
    # else:
    #     plot_ions = False
    #     n_subplots=2
    #     title_snippet = 'EPD electrons'
    #     labels = [
    #         'Omnidirectional electrons',
    #         'Electron precipitation ratio',
    #         ]
        
_epd_epad = EPD_PAD(sc_id, time_range, start_pa=0, min_counts=0, accumulate=1, lc_exclusion_angle=0)

fig, ax = plt.subplots(n_subplots, 1, sharex=True, figsize=(8, 7))
_cbars = [None]*n_subplots

p, _cbars[0] = _epd_epad.plot_omni(ax[0], vmin=1E2, vmax=1E7, pretty_plot=False)
p, _cbars[1] = _epd_epad.plot_blc_dlc_ratio(ax[1], vmin=1E-2, vmax=1)

ax[1].contour(
    _epd_epad.pad.time, 
    _epd_epad.energy, 
    (_epd_epad.blc/_epd_epad.dlc).T,
    levels=[0.8], 
    colors='k'
)

if plot_ions:
    p, _cbars[2] = _epd_ipad.plot_omni(ax[2], vmin=1E2, vmax=1E7, pretty_plot=False)
    p, _cbars[3] = _epd_ipad.plot_blc_dlc_ratio(ax[3], vmin=1E-2, vmax=1)

    ax[3].contour(
        _epd_ipad.counts['time'], 
        _epd_ipad.counts['energy'], 
        (_epd_ipad.blc/_epd_ipad.dlc).T,
        levels=[0.8], 
        colors='k'
    )
_epd_epad.plot_position(ax[-1])

for ax_i in ax:
    ax_i.set(ylim=(60, None))

ax[0].set_title(
    f'ELFIN-{sc_id.upper()} | {title_snippet} | {_epd_epad.time_range[0]:%Y-%m-%d}'
    )
plt.subplots_adjust(bottom=0.2, right=1, top=0.90, hspace=0.034)

for (ax_i, panel_letter, label, cbar) in zip(ax, string.ascii_lowercase, labels, _cbars):
    _text = ax_i.text(
        0.01, 0.99, f'({panel_letter}) {label}', va='top', transform=ax_i.transAxes, 
        fontsize=12
        )
    _text.set_bbox(dict(facecolor='grey', linewidth=0, pad=0.1, edgecolor='k'))
    ax_i.set_facecolor='grey'
    cbar.set_label(cbar.ax.yaxis.label._text, fontsize=12)

ax[-1].xaxis.set_label_coords(-0.1, -0.01*(1+n_subplots))
ax[-1].tick_params(axis='x', labelsize=10)

time_range = pd.to_datetime(time_range)
save_name = (
            f'{time_range[0]:%Y%m%d_%H%M%S}_'
            f'{time_range[1]:%H%M%S}_'
            f'elfin{sc_id.lower()}_e_ib'
            )
# plt.savefig((elfinasi.plot_dir / save_name).with_suffix('.png'))
# plt.savefig((elfinasi.plot_dir / save_name).with_suffix('.pdf'))
plt.show()