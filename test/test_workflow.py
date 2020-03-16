#%matplotlib qt
import mocalum as mc


tmp = mc.Mocalum()
tmp.set_ivap_probing([0,0,0], 30, 0, 1, 5, 1000, 10000)
tmp.gen_unc_contributors()
tmp.gen_plaw_ffield()
tmp.calc_los_speed()
tmp.rc_wind()
#tmp.data.probing.x.plot()