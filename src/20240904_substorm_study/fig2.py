"""
Make a THEMIS and GOES summary plots.

THEMIS variable descriptions: http://themis.ssl.berkeley.edu//themisftp/3%20Ground%20Systems/3.2%20Science%20Operations/Science%20Operations%20Documents/Science%20Data%20Variable%20Descriptions/THEMIS%20Science%20Data%20Variables%20Descriptions.pdf
"""
import dateutil.parser
import pathlib
from datetime import datetime
import string

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import matplotlib.colors
import matplotlib.dates
import pyspedas
import pytplot
import manylabels
import pandas as pd
import numpy as np

import asilib.asi

save_plots = True

ELECTRON_CHARGE = 1.60217663E-19  # Coulombs
ELECTRON_MASS = 9.1093837E-31  # kg
PERMITTIVITY = 8.8541878188E-12	 # C^2⋅kg^−1⋅m^−3⋅s^2
SPEED_OF_LIGHT = 3E8 # ms/s
R_E = 6378.137  # km


time_range = ('2022-09-04T04:00', '2022-09-04T04:30')
_time_range = [dateutil.parser.parse(t_i) for t_i in time_range]
themis_probes = ('a', 'd', 'e')
sst_bounds = np.array([
    [2E5, 3E6],
    [1E4, 1E6],
    [1E4, 3E6],
])
themis_Ew = ('fff_32_edc34', 'fff_32_edc34', 'fff_32_edc12')
themis_Bw = ('fff_32_scm3', 'fff_32_scm3', 'fff_32_scm3')
coordinates = 'gsm'
locations={}
_b_labels = (f'$B_{{x}}$', f'$B_{{y}}$', f'$B_{{z}}$', f'$|B|$')
_colors = ('#1f77b4', '#ff7f0e', '#2ca02c', 'k')
_line_styles = ('-', '-', '-', 'dashdot')
_subplot_labels = (
    'DC Magnetic Field',
    'DC magnetic field inclination', 
    'Ion Flows', 
    'Energetic Electrons', 
    'Low-energy Electrons',
    'Magnetic Field Waves',
    'Electric Field Waves',
    'TREx-RGB GILL & PINA keogram'
    )
text_ylocs = 0.94*np.ones(len(_subplot_labels))
text_ylocs[1] = 0.25

vertical_lines = (
    datetime(2022, 9, 4, 4, 17, 0),
    datetime(2022, 9, 4, 4, 20, 0)
)

_bmag_lims = [[-25, 70], [None, None], [None, None]]


gill_asi = asilib.asi.trex_rgb('gill', time_range=time_range)
pina_asi = asilib.asi.trex_rgb('pina', time_range=time_range)

for i, (probe, _bmag_lim) in enumerate(zip(themis_probes, _bmag_lims)):
    fgm_vars = pyspedas.themis.fgm(probe=probe, trange=time_range, time_clip=True, no_update=True)
    sst_vars = pyspedas.themis.sst(probe=probe, trange=time_range, time_clip=True, no_update=True)
    mom_vars = pyspedas.themis.mom(probe=probe, trange=time_range, time_clip=True, no_update=True)
    esa_vars = pyspedas.themis.esa(probe=probe, trange=time_range, time_clip=True, no_update=True)
    fft_vars = pyspedas.themis.fft(probe=probe, trange=time_range, time_clip=True, no_update=True)
    state_vars = pyspedas.themis.state(probe=probe, trange=time_range, time_clip=True, no_update=True)

    state_xr = pytplot.get_data(f'th{probe}_pos_{coordinates}')
    state_times = pyspedas.time_datetime(state_xr.times)
    state_times = [state_time.replace(tzinfo=None) for state_time in state_times]
    state_df = pd.DataFrame(
        index=state_times,
        data={f'{component.upper()} [Re]':state_xr.y[:, i]/R_E for i, component in enumerate(['x', 'y', 'z'])}
        )
    state_df.insert(0, 'R', np.linalg.norm(state_xr.y/R_E, axis=1))

    fig, ax = plt.subplots(len(_subplot_labels), sharex=True, figsize=(10, 10))

    fgm_xr = pytplot.get_data(f'th{probe}_fgl_{coordinates}')
    mag_data = np.concatenate((fgm_xr.y, np.linalg.norm(fgm_xr.y, axis=1)[..., np.newaxis]), axis=1)
    mag_times = pyspedas.time_datetime(fgm_xr.times)

    for mag_component, color, label, _line_style in zip(mag_data.T, _colors, _b_labels, _line_styles):
        ax[0].plot(mag_times, mag_component, color=color, label=label, ls=_line_style)
    ax[0].legend(
        loc='lower left',
        bbox_to_anchor=(1.01, 0.01)
        )
    ax[0].axhline(0, c='k', ls='--')
    ax[0].set(
        ylabel=f'B [nT]',
        xlim=_time_range,
        ylim=_bmag_lim
        )
    
    magnetic_inclination = np.abs(fgm_xr.y[:, 2])/np.linalg.norm(fgm_xr.y[:, 1:], axis=1)
    ax[1].plot(mag_times, magnetic_inclination, color='k')
    ax[1].set(
        ylabel=f'$|B_z|/\sqrt{{B_x^2 + B_y^2}}$',
        ylim=(0, 1.1)
        )

    vi_xr = pytplot.get_data(f'th{probe}_peim_velocity_{coordinates}')
    _lines = ax[2].plot(pyspedas.time_datetime(vi_xr.times), vi_xr.y)
    ax[2].legend(
        iter(_lines), 
        [f'$v_{{{i}}}$' for i in [f'x', 'y', 'z']], 
        loc='lower left',
        bbox_to_anchor=(1.01, 0.01)
        )
    ax[2].axhline(0, c='k', ls='--')
    ax[2].set(ylim=(-100, 100), ylabel=f'$v_{{i}}$ [km/s]')

    sst_xr = pytplot.get_data(f'th{probe}_psef_en_eflux')
    valid_e_channels = np.where(~np.isnan(sst_xr.v[0, :]))[0]
    p = ax[3].pcolormesh(
        np.array(pyspedas.time_datetime(sst_xr.times)),
        sst_xr.v[0, valid_e_channels]/1E3,  # Convert to keV units
        sst_xr.y[:, valid_e_channels].T,
        norm=matplotlib.colors.LogNorm(vmin=sst_bounds[i, 0], vmax=sst_bounds[i, 1]),
        rasterized=True
        )
    ax[3].set(yscale='log', ylabel='Energy [keV]')
    cbar = plt.colorbar(p, ax=ax[3])
    cbar.set_label(f'Electron flux\n$[eV/(cm^{{2}} \ s \ sr \ eV)]$', y=0, ha='center')

    esa_spectra = pytplot.get_data('tha_peef_en_eflux', xarray=True)
    idx = np.where(~np.isnan(esa_spectra.spec_bins[0,:]))[0]
    p = ax[4].pcolormesh(
        esa_spectra.time, 
        esa_spectra.spec_bins[0, idx]/1E3, 
        esa_spectra.values[:, idx].T, 
        norm=matplotlib.colors.LogNorm()
        )
    ax[4].set(yscale='log', ylabel='Energy [keV]')
    plt.colorbar(p, ax=ax[4])

    bw_xr = pytplot.get_data(f'th{probe}_{themis_Bw[i]}')
    p = ax[5].pcolormesh(
        np.array(pyspedas.time_datetime(bw_xr.times)), 
        bw_xr.v, 
        bw_xr.y.T, 
        norm=matplotlib.colors.LogNorm(vmax=1E-5),
        shading='nearest',
        rasterized=True
        )
    plt.colorbar(p, ax=ax[5], label=f'$nT^{{2}}/Hz$')
    ax[5].set(yscale='log', ylabel=f'$B_{{w}}$ [Hz]')
    f_ce = np.abs(ELECTRON_CHARGE)*1E-9*np.linalg.norm(fgm_xr.y, axis=1)/(2*np.pi*ELECTRON_MASS)

    ew_xr = pytplot.get_data(f'th{probe}_{themis_Ew[i]}')
    p = ax[6].pcolormesh(
        pyspedas.time_datetime(ew_xr.times), 
        ew_xr.v, 
        ew_xr.y.T, 
        norm=matplotlib.colors.LogNorm(vmax=1E-7),
        rasterized=True
        )
    plt.colorbar(p, ax=ax[6], label=f'$(mV/m)^{{2}}/Hz$')
    ax[6].set(yscale='log', ylabel=f'$E_{{w}}$ [Hz]')

    for ax_i in ax[[5,6]]:
        ax_i.plot(pyspedas.time_datetime(fgm_xr.times), f_ce, label=f'$f_{{ce}}$', c='w', ls='-')
        ax_i.plot(pyspedas.time_datetime(fgm_xr.times), f_ce/2, label=f'$f_{{ce}}/2$', c='w', ls='--')
        ax_i.plot(pyspedas.time_datetime(fgm_xr.times), f_ce/10, label=f'$f_{{ce}}/10$', c='w', ls=':')
        ax_i.set_facecolor('k')
    ax[5].legend(loc='upper right', facecolor='grey', labelcolor='white')

    # Plot keogram along THEMIS's latitude.
    _themis_lon = -98.2  # By eye---too lazy to copy the mapping code here.
    gill_keogram_path=np.stack((np.linspace(55, 60, num=500), _themis_lon*np.ones(500)), axis=1)
    pina_keogram_path=np.stack((np.linspace(45, 55.5, num=500), _themis_lon*np.ones(500)), axis=1)
    gill_asi.plot_keogram(ax=ax[7], aacgm=True, title=False, path=gill_keogram_path, pcolormesh_kwargs={'rasterized':True})
    pina_asi.plot_keogram(ax=ax[7], aacgm=True, title=False, path=pina_keogram_path, pcolormesh_kwargs={'rasterized':True})
    ax[7].set(ylabel=f'AACGM ${{\lambda}}$ [${{\circ}}$]')

    plt.suptitle(f'THEMIS-{probe.upper()} | {time_range[0][:10]} | {coordinates.upper()} vector coordinates')
    manylabels.ManyLabels(ax[-1], state_df)

    for ax_i, letter, _label, text_yloc in zip(ax, string.ascii_lowercase, _subplot_labels, text_ylocs):
        for vertical_line in vertical_lines:
            ax_i.axvline(vertical_line, c='grey', ls='--')
        _t = ax_i.text(
            0.01, text_yloc, f'({letter}) {_label}', transform=ax_i.transAxes, fontsize=12, 
            va='top', 
            color='white'
            )
        _t.set_bbox(dict(facecolor='grey', alpha=0.5))
        ax_i.xaxis.set_minor_locator(matplotlib.dates.MinuteLocator())

    for ax_i in ax[[0, 1, 2, 7]]:
        divider = make_axes_locatable(ax_i)
        cax = divider.append_axes("right", size="24%", pad=0.08)
        cax.remove()

    plt.subplots_adjust(top=0.959, bottom=0.112, left=0.118, right=0.98, hspace=0.1, wspace=0.1)
    if save_plots:
        (pathlib.Path(__file__).parent / 'plots').mkdir(exist_ok=True)
        file_name = f'{_time_range[0]:%Y%m%d_%H%M}_{_time_range[1]:%H%M}_themis{probe.lower()}_summary'
        plt.savefig(pathlib.Path(pathlib.Path(__file__).parent / 'plots' / file_name).with_suffix('.png'))
        plt.savefig(pathlib.Path(pathlib.Path(__file__).parent / 'plots' / file_name).with_suffix('.pdf'))
    else:
        plt.show()
        break