import matplotlib.testing.decorators
import matplotlib.pyplot as plt
from matplotlib.dates import num2date

from elfinasi.elfin import EPD_PAD

@matplotlib.testing.decorators.image_comparison(
    baseline_images=['test_pad_plot'],
    tol=10,
    remove_text=True,
    extensions=['png'],
)
def test_pad_plot():
    time_range = ('2020-09-02T14:20', '2020-09-02T14:25')
    sc_id = 'B'
    pad_obj = EPD_PAD(
        sc_id, time_range, start_pa=0, min_counts=None, spin_time_tol=(2.5, 3.5), accumulate=1
        )
        
    fig, ax = plt.subplots(7, 1, sharex=True, figsize=(7, 8))
    ax[0].set_title(
        f'ELFIN-{sc_id.upper()} | {time_range[0]}-{time_range[1]}'
        f'\nElectron Pitch Angle Distributions'
        )
    pad_obj.plot_omni(ax[0])
    pad_obj.plot_pad_scatter(ax[1], energy=520)
    pad_obj.plot_pad_spectrogram(ax[2], energy=63)
    pad_obj.plot_pad_spectrogram(ax[3], energy=138)
    pad_obj.plot_pad_spectrogram(ax[4], energy=305)
    pad_obj.plot_pad_spectrogram(ax[5], energy=520)
    pad_obj.plot_blc_dlc_ratio(ax[-1], vmax=1, cmap='viridis')
    pad_obj.plot_position(ax[-1])
    ax[-1].set_xlim(*pad_obj.time_range)

    for ax_i in ax:
        ax_i.fmt_xdata = lambda x: num2date(x).replace(tzinfo=None).isoformat()
    plt.subplots_adjust(bottom=0.168, right=0.927, top=0.948, hspace=0.133)


@matplotlib.testing.decorators.image_comparison(
    baseline_images=['test_pad_plot_min_counts'],
    tol=10,
    remove_text=True,
    extensions=['png'],
)
def test_pad_plot_min_counts():
    time_range = ('2020-09-02T14:20', '2020-09-02T14:25')
    sc_id = 'B'
    pad_obj = EPD_PAD(
        sc_id, time_range, start_pa=0, min_counts=10, spin_time_tol=(2.5, 3.5), accumulate=1
        )
        
    fig, ax = plt.subplots(7, 1, sharex=True, figsize=(7, 8))
    ax[0].set_title(
        f'ELFIN-{sc_id.upper()} | {time_range[0]}-{time_range[1]}'
        f'\nElectron Pitch Angle Distributions'
        )
    pad_obj.plot_omni(ax[0])
    pad_obj.plot_pad_scatter(ax[1], energy=520)
    pad_obj.plot_pad_spectrogram(ax[2], energy=63)
    pad_obj.plot_pad_spectrogram(ax[3], energy=138)
    pad_obj.plot_pad_spectrogram(ax[4], energy=305)
    pad_obj.plot_pad_spectrogram(ax[5], energy=520)
    pad_obj.plot_blc_dlc_ratio(ax[-1], vmax=1, cmap='viridis')
    pad_obj.plot_position(ax[-1])
    ax[-1].set_xlim(*pad_obj.time_range)

    for ax_i in ax:
        ax_i.fmt_xdata = lambda x: num2date(x).replace(tzinfo=None).isoformat()
    plt.subplots_adjust(bottom=0.168, right=0.927, top=0.948, hspace=0.133)


@matplotlib.testing.decorators.image_comparison(
    baseline_images=['test_pad_count_plot'],
    tol=10,
    remove_text=True,
    extensions=['png'],
)
def test_pad_count_plot():
    plot_flux = False
    vmin=1

    time_range = ('2021-07-31T21:09', '2021-07-31T21:11')
    sc_id = 'A'
    pad_obj = EPD_PAD(
        sc_id, time_range, start_pa=0, min_counts=None, spin_time_tol=(2.5, 3.5), accumulate=1
        )
        
    fig, ax = plt.subplots(7, 1, sharex=True, figsize=(7, 8))
    ax[0].set_title(
        f'ELFIN-{sc_id.upper()} | {time_range[0]}-{time_range[1]}'
        f'\nElectron Pitch Angle Distributions'
        )
    pad_obj.plot_omni(ax[0], flux=plot_flux, vmin=vmin)
    pad_obj.plot_pad_scatter(ax[1], flux=plot_flux, vmin=vmin, energy=520)
    pad_obj.plot_pad_spectrogram(ax[2], flux=plot_flux, vmin=vmin, energy=63)
    pad_obj.plot_pad_spectrogram(ax[3], flux=plot_flux, vmin=vmin, energy=138)
    pad_obj.plot_pad_spectrogram(ax[4], flux=plot_flux, vmin=vmin, energy=305)
    pad_obj.plot_pad_spectrogram(ax[5], flux=plot_flux, vmin=vmin, energy=520)
    pad_obj.plot_blc_dlc_ratio(ax[-1], vmax=1, cmap='viridis')
    pad_obj.plot_position(ax[-1])
    ax[-1].set_xlim(*pad_obj.time_range)

    for ax_i in ax:
        ax_i.fmt_xdata = lambda x: num2date(x).replace(tzinfo=None).isoformat()
    plt.subplots_adjust(bottom=0.168, right=0.927, top=0.948, hspace=0.133)