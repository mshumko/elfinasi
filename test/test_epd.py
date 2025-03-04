"""
Tests to ensure that the ELFIN-EPD electron and ion data loads correctly.
"""
from datetime import datetime

from elfinasi.elfin import EPD

def test_load_epd_l1_electrons():
    sc_id = 'B'
    day = datetime(2022, 8, 10)
    elfin_epd = EPD(sc_id, day, species='electron', level=1)
    assert elfin_epd.file_path.name == 'elb_l1_epdef_20220810_v01.cdf'
    assert elfin_epd.keys() == [
        'elb_pef_time', 
        'elb_pef', 
        'elb_pef_energies_mean', 
        'elb_pef_energies_min', 
        'elb_pef_energies_max', 
        'elb_pef_sectnum', 
        'elb_pef_nspinsinsum', 
        'elb_pef_nsectors', 
        'elb_pef_spinper'
        ]
    assert str(elfin_epd['epoch'][0]) == '2022-08-10 01:41:37.968884'
    assert str(elfin_epd['epoch'][-1]) == '2022-08-10 16:56:41.615771'
    assert elfin_epd['epoch'].shape == (5712,)
    return

def test_load_epd_l2_electrons():
    sc_id = 'A'
    day = datetime(2022, 8, 10)
    elfin_epd = EPD(sc_id, day, species='electron', level=2)
    assert elfin_epd.file_path.name == 'ela_l2_epdef_20220810_v01.cdf'
    assert elfin_epd.keys() == [
        'ela_pef_et_time', 
        'ela_pef_Et_nflux', 
        'ela_pef_Et_eflux', 
        'ela_pef_Et_dfovf', 
        'ela_pef_energies_mean', 
        'ela_pef_energies_min', 
        'ela_pef_energies_max', 
        'ela_pef_pa', 
        'ela_pef_spinphase', 
        'ela_pef_sectnum', 
        'ela_pef_Tspin', 
        'ela_pef_hs_time', 
        'ela_pef_hs_Epat_nflux', 
        'ela_pef_hs_Epat_eflux', 
        'ela_pef_hs_Epat_dfovf', 
        'ela_pef_hs_LCdeg', 
        'ela_pef_hs_antiLCdeg', 
        'ela_pef_hs_epa_spec', 
        'ela_pef_fs_time', 
        'ela_pef_fs_Epat_nflux', 
        'ela_pef_fs_Epat_eflux', 
        'ela_pef_fs_Epat_dfovf', 
        'ela_pef_fs_LCdeg', 
        'ela_pef_fs_antiLCdeg', 
        'ela_pef_fs_epa_spec', 
        'ela_pef_nspinsinsum', 
        'ela_pef_nsectors', 
        'ela_pef_sect2add', 
        'ela_pef_spinph2add'
        ]
    assert str(elfin_epd['ela_pef_et_time'][0]) == '2022-08-10 02:04:35.598856'
    assert str(elfin_epd['ela_pef_et_time'][-1]) == '2022-08-10 17:16:53.588310'
    assert elfin_epd['ela_pef_et_time'].shape == (4192,)
    return

def test_load_epd_l1_ions():
    sc_id = 'B'
    day = '2022-08-10'
    elfin_epd = EPD(sc_id, day, species='ion', level=1)
    assert elfin_epd.file_path.name == 'elb_l1_epdif_20220810_v01.cdf'
    assert elfin_epd.keys() == [
        'elb_pif_time', 
        'elb_pif', 
        'elb_pif_energies_mean', 
        'elb_pif_energies_min', 
        'elb_pif_energies_max', 
        'elb_pif_sectnum', 
        'elb_pif_nspinsinsum', 
        'elb_pif_nsectors', 
        'elb_pif_spinper', 
        'elb_pif_spinphase'
        ]
    assert str(elfin_epd['epoch'][0]) == '2022-08-10 01:41:37.968884'
    assert str(elfin_epd['epoch'][-1]) == '2022-08-10 16:56:41.615771'
    return