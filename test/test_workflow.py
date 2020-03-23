#%matplotlib qt
import mocalum as mc
import matplotlib.pyplot as plt



tmp = mc.Mocalum()
tmp.set_ivap_probing([0,0,0], 30, 0, 1, 5, 1000, 10000)
tmp.gen_unc_contributors()
tmp.gen_plaw_ffield()
tmp.calc_los_speed()
tmp.reconstruct_wind()
tmp.data.rc_wind.ws.plot()
plt.show()