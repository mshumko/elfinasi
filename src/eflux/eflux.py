import string
import dateutil.parser

import matplotlib.transforms
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import matplotlib.colors
import numpy as np
import pandas as pd


import elfinasi

events = [
    {
        'time_range':('2022-09-04T04:18:00', '2022-09-04T04:23:00'),
        'sc_id':'A',
        'location_codes':('ATHA', 'PINA', 'GILL', 'RABB', 'LUCK'),
    },
]

kev_erg_factor = 1.6E-9  # The conversion factor from KeV to ergs.
precipitation_solid_angle = 2*np.pi


def load_gabrielse_data(path):
    header = [
        "date/time", 
        "ELFINLat", 
        "PINALat", 
        "GILLlat", 
        "EFLINLon", 
        "PINALon", 
        "GILLlon", 
        "ClosestStation", 
        "Elevation [deg]",
        "ELFINeflux [erg/cm^2/s]", 
        "ELFINenergy [keV]"
        ]
    themis_asi_eflux = pd.read_csv(
        path, 
        index_col=0, 
        skiprows=1,  # This is necessary because someone forgot to add ","s to the header!
        names=header,
        parse_dates=True, 
        na_values=('NaN', "-1")
    )
    themis_asi_eflux = themis_asi_eflux.loc[themis_asi_eflux['Elevation [deg]'] > 10]  # Christine's suggested threshold
    themis_asi_eflux.index = themis_asi_eflux.index.astype('datetime64[ms]')
    return themis_asi_eflux

for event in events:
    pad_obj_eflux = elfinasi.EPD_PAD(
        event['sc_id'], 
        event['time_range'], 
        start_pa=0, 
        min_counts=None, 
        accumulate=1, 
        spin_time_tol=(2.5, 12),
        lc_exclusion_angle=0, 
        nflux=False
    )
    _date = dateutil.parser.parse(event['time_range'][0]).strftime('%Y%m%d')
    themis_asi_eflux = load_gabrielse_data(elfinasi.data_dir / f'{_date}_themis_asi_inversion' / 'ASIdataCSV0.csv')

    energy_widths_mev = (pad_obj_eflux.energy_widths[:, 1]-pad_obj_eflux.energy_widths[:, 0])/1E3
    eflux_ergs = kev_erg_factor*precipitation_solid_angle*(pad_obj_eflux.blc - pad_obj_eflux.ablc)*energy_widths_mev
    relativistic_eflux = np.nansum(eflux_ergs, axis=1)
    elfin_eflux = pd.DataFrame(relativistic_eflux, index=pad_obj_eflux.pad.time.astype('datetime64[ms]'), columns=['energetic'])
    auroral_eflux = pd.DataFrame(themis_asi_eflux['ELFINeflux [erg/cm^2/s]']).rename(columns={'ELFINeflux [erg/cm^2/s]':'auroral'})

    merged_eflux = pd.merge_asof(
        elfin_eflux, 
        auroral_eflux, 
        left_index=True, 
        right_index=True, 
        direction='nearest', 
        tolerance=pd.Timedelta('1s')
        )
    merged_eflux['energetic_contribution'] = 100*merged_eflux['energetic']/(merged_eflux['auroral'] + merged_eflux['energetic'])

    fig, bx = plt.subplots(3, 1, sharex=True, figsize=(8, 6))
    pad_obj_eflux.plot_omni(bx[0], labels=True, colorbar=True, vmin=1E2, vmax=1E9, pretty_plot=False, fraction=0.05)
    bx[1].plot(
        merged_eflux.index, 
        merged_eflux['energetic'], 
        label='$Q_{{>50 \\ \\mathrm{{keV}}}}$', 
        color='r', 
        linestyle='--'
        )
    bx[1].plot(
        merged_eflux.index, 
        merged_eflux['auroral'], 
        label=f'$Q_{{\\mathrm{{ASI}}}}$', 
        color='k'
        )
    bx[1].legend(loc='upper right', fontsize=10)

    bx[2].plot(merged_eflux.index, merged_eflux['energetic_contribution'], color='k')
    pad_obj_eflux.plot_position(bx[-1])
    bx[-1].xaxis.set_major_locator(plt.MaxNLocator(7))
    bx[-1].xaxis.set_label_coords(-0.04, -0.007*7)
    bx[-1].xaxis.label.set_size(10)

    for bx_i in bx[[1, 2]]:
        divider = make_axes_locatable(bx_i)
        cax = divider.append_axes("right", size="10%", pad=0.08)
        cax.remove()

    bx[1].set_yscale('log')
    bx[1].set_ylabel(f'Energy Flux\n$[ergs/cm^{{2}}s]$')
    bx[1].set_yticks([1E-2, 1E-1, 1E0, 1E1])
    bx[1].set_ylim(1E-3, 1E2)

    bx[2].set_ylabel(f'Percentage')
    bx[2].set_ylim(0, 1E2)
    bx[2].axhline(50, color='k', linestyle='--')
    # bx[2].text(0.98, 
    #            51, 
    #            f'$Q_{{>50 \\ \\mathrm{{keV}}}} and Q_{{\\mathrm{{ASI}}}}$ equal',
    #            ha='right',
    #            va='bottom',
    #            transform=matplotlib.transforms.blended_transform_factory(
    #                bx[2].transAxes, bx[2].transData
    #                )
    # )

    labels = (
        'ELFIN Differential Q', 
        f'Q along ELFIN track', 
        f'$Q_{{>50 \\ \\mathrm{{keV}}}}/(Q_{{>50 \\ \\mathrm{{keV}}}} + Q_{{\\mathrm{{ASI}}}})$'
        )
    for ax_i, label, letter in zip(bx, labels, string.ascii_lowercase):
        _text = ax_i.text(0.01, 0.99, f'({letter}) {label}', transform=ax_i.transAxes, va='top')
        _text.set_bbox(dict(facecolor='white', linewidth=0, pad=0.1, edgecolor='k'))
    plt.suptitle(f'ELFIN-{event["sc_id"].upper()} - THEMIS ASI Electron Flux Comparison', fontsize=14)
    plt.tight_layout()
    plt.show()