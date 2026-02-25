import string
import dateutil.parser
import warnings

import matplotlib.transforms
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import matplotlib.colors
import numpy as np
import pandas as pd
import asilib.asi
import asilib.map
import cartopy.crs as ccrs

import elfinasi
from elfinasi import map_elfin

events = [
    {
        'time_range':('2022-09-04T04:18:00', '2022-09-04T04:23:00'),
        'sc_id':'A',
        'location_codes':('PINA', 'GILL', 'TPAS', 'KAPU'),
        'array':asilib.asi.themis,
    },
    {
        'time_range':('2021/08/31 06:04', '2021/08/31 06:10'),
        'sc_id':'B',
        'location_codes':('PINA', 'GILL'),  # Can also add 'RABB', but it was partly cloudy.
        'array':asilib.asi.trex_rgb,
    },
    {
        'time_range':('2021/08/31 06:04', '2021/08/31 06:10'),
        'sc_id':'A',
        'location_codes':('PINA', 'GILL'),  # Can also add 'RABB', but it was partly cloudy.
        'array':asilib.asi.trex_rgb,
    },
]

kev_erg_factor = 1.6E-9  # The conversion factor from KeV to ergs.
precipitation_solid_angle = 2*np.pi
alt=110
n_images = 4

labels = (
        'ELFIN Differential Q', 
        f'Q along ELFIN track', 
        f'$Q_{{>50 \\ \\mathrm{{keV}}}}/(Q_{{>50 \\ \\mathrm{{keV}}}} + Q_{{\\mathrm{{ASI}}}})$'
        )


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
    image_times = pd.date_range(event['time_range'][0], event['time_range'][1], periods=n_images+2)[1:-1]
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
    transformed_state = pad_obj_eflux.transform_state()
    transformed_state = transformed_state.loc[event['time_range'][0]:event['time_range'][1]]
    mapped_state = map_elfin(transformed_state, alt=alt)

    _date = dateutil.parser.parse(event['time_range'][0]).strftime('%Y%m%d')
    if not (elfinasi.data_dir / f'{_date}_themis_asi_inversion' / 'ASIdataCSV0.csv').exists():
        warnings.warn(f"Could not find THEMIS ASI inversion data for {event['time_range'][0][:10]}.")
        inversion_data_loaded = False
    else:
        themis_asi_eflux = load_gabrielse_data(elfinasi.data_dir / f'{_date}_themis_asi_inversion' / 'ASIdataCSV0.csv')
        inversion_data_loaded = True

    energy_widths_mev = (pad_obj_eflux.energy_widths[:, 1]-pad_obj_eflux.energy_widths[:, 0])/1E3
    eflux_ergs = kev_erg_factor*precipitation_solid_angle*(pad_obj_eflux.blc - pad_obj_eflux.ablc)*energy_widths_mev
    relativistic_eflux = np.nansum(eflux_ergs, axis=1)
    elfin_eflux = pd.DataFrame(relativistic_eflux, index=pad_obj_eflux.pad.time.astype('datetime64[ms]'), columns=['energetic'])

    if inversion_data_loaded:
        auroral_eflux = pd.DataFrame(
            themis_asi_eflux['ELFINeflux [erg/cm^2/s]']
            ).rename(columns={'ELFINeflux [erg/cm^2/s]':'auroral'})

        merged_eflux = pd.merge_asof(
            elfin_eflux, 
            auroral_eflux, 
            left_index=True, 
            right_index=True, 
            direction='nearest', 
            tolerance=pd.Timedelta('1s')
            )
        
        merged_eflux['energetic_contribution'] = 100*merged_eflux['energetic']/(merged_eflux['auroral'] + merged_eflux['energetic'])
    else:
        merged_eflux = elfin_eflux.copy()
        merged_eflux['auroral'] = np.nan
        merged_eflux['energetic_contribution'] = np.nan

    fig = plt.figure(figsize=(9, 7.5))
    # Add a gridspec with two rows and two columns and a ratio of 1 to 4 between
    # the size of the marginal Axes and the main Axes in both directions.
    # Also adjust the subplot parameters for a square plot.
    outer_gridspec = fig.add_gridspec(
        2, 
        1,
        left=0.03, 
        right=0.95, 
        bottom=0.20, 
        top=0.95,
        hspace=0.1,
        height_ratios=(0.75, 1)
        )
    top_gs = matplotlib.gridspec.GridSpecFromSubplotSpec(
        1, 
        len(image_times), 
        subplot_spec=outer_gridspec[0, 0],
        wspace=0.02
        )
    bottom_gs = matplotlib.gridspec.GridSpecFromSubplotSpec(
        len(labels), 
        10, 
        subplot_spec=outer_gridspec[1, 0], 
        hspace=0.05,
        )
    
    asis = asilib.Imagers(
        [event['array'](location_code=location_code, time_range=event['time_range'], alt=alt) 
        for location_code in event['location_codes']]
        )

    ax = [
        asilib.map.create_map(
            lon_bounds=(np.mean(asis.lon_bounds)-10, np.mean(asis.lon_bounds)+10), 
            lat_bounds=asis.lat_bounds, 
            fig_ax=(fig, top_gs[0, i]), 
            land_color='grey'
            ) for i in range(top_gs.ncols)
        ]
    bx = [None]*len(labels)
    bx[0] = fig.add_subplot(bottom_gs[0, 1:])
    for i in range(1, len(labels)):
        bx[i] = fig.add_subplot(bottom_gs[i, 1:], sharex=bx[0])

    for time, ax_i, _label in zip(image_times, ax, string.ascii_lowercase):
        asis = asilib.Imagers(
            [event['array'](location_code=location_code, time=time, alt=alt) 
            for location_code in event['location_codes']]
            )
        for _imager in asis.imagers:
            _imager.set_color_bounds(*_imager.auto_color_bounds())
        # vmin = min([_imager.get_color_bounds()[0] for _imager in asis.imagers])
        # vmax = max([_imager.get_color_bounds()[1] for _imager in asis.imagers])
        asis.plot_map(
            ax=ax_i, 
            min_elevation=10, 
            pcolormesh_kwargs={'rasterized':True}, 
            asi_label=True, 
            )
        ax_i.plot(mapped_state['lon'], mapped_state['lat'], c='r', transform=ccrs.PlateCarree())
        ax_i.scatter(
            mapped_state.loc[time, 'lon'], 
            mapped_state.loc[time, 'lat'], 
            c='r', 
            s=30,
            transform=ccrs.PlateCarree(),
            label=f'ELFIN-{event["sc_id"].upper()}'
            )
        _text=ax_i.text(
            0.01, 0.99, f'({_label}) {time:%H:%M:%S}', va='top', transform=ax_i.transAxes, fontsize=12
            )
        _text.set_bbox(dict(facecolor='white', linewidth=0, pad=0.1, edgecolor='k'))

    ax[0].legend(loc='lower left', ncols=2, columnspacing=0.1, handletextpad=0.1, fontsize=8)

    for bx_i in bx[:-1]:
        bx_i.get_xaxis().set_visible(False)

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

    for bx_i in bx[1:]:
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

    # Connect the subplots and add vertical lines to cx and dx.
    for ax_i, image_time_numeric in zip(ax, matplotlib.dates.date2num(image_times)):
        line = matplotlib.patches.ConnectionPatch(
            xyA=(0.5, 0), coordsA=ax_i.transAxes,
            xyB=(image_time_numeric, bx[0].get_ylim()[1]), coordsB=bx[0].transData, 
            ls='--')
        ax_i.add_artist(line)

        for _other_ax in bx:
            _other_ax.axvline(image_time_numeric, c='k', ls='--', alpha=1)

    for ax_i, label, letter in zip(bx, labels, string.ascii_lowercase[len(ax):]):
        _text = ax_i.text(0.01, 0.99, f'({letter}) {label}', transform=ax_i.transAxes, va='top', fontsize=12)
        _text.set_bbox(dict(facecolor='white', linewidth=0, pad=0.1, edgecolor='k'))

    pad_obj_eflux.plot_position(bx[-1])
    bx[-1].xaxis.set_major_locator(plt.MaxNLocator(7))
    bx[-1].xaxis.set_label_coords(-0.07, -0.007*7)
    bx[-1].xaxis.label.set_size(9)
    plt.suptitle(f'ELFIN-{event["sc_id"].upper()} - THEMIS ASI Electron Flux Comparison', fontsize=14)
    plt.tight_layout()
    file_name = (
        f'{dateutil.parser.parse(event["time_range"][0]).strftime("%Y%m%d_%H%M")}_'
        f'{dateutil.parser.parse(event["time_range"][1]).strftime("%H%M")}_'
        f'elfin{event["sc_id"].lower()}_themisasi_eflux_comparison')
    for ext in ['png', 'pdf']:
        plt.savefig(elfinasi.plot_dir / 'eflux' / f'{file_name}.{ext}', dpi=300)
    # plt.show()