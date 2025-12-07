import sys

from tools.obench2 import read_obench_file

from rayoptics.environment import *
from tools.helpers import spot_analysis, plot_spot_to_file, ray_abr_analysis, opd_analysis, plot_ray_abr_to_file, \
    multiplot_spot_to_file
import matplotlib.pyplot as plt
from rayoptics.raytr.trace import apply_paraxial_vignetting

if len(sys.argv) > 1:
    arg = sys.argv[1]
else:
    print("No argument provided.")
    quit(1)

opm,dict = read_obench_file(arg)

osp = opm.optical_spec
sm = opm.seq_model
sm.list_surfaces()
sm.list_gaps()
sm.do_apertures = False
opm.update_model()
apply_paraxial_vignetting(opm)
sm.list_model()
listobj(osp)

layout_plt = plt.figure(FigureClass=InteractiveLayout, opt_model=opm, do_draw_rays=True, do_paraxial_layout=False,
                        is_dark=True).plot()
layout_plt.savefig("layout.svg", format="svg")

spot_results = spot_analysis(opm,num_rings=21,apply_vignetting=True)
plot_spot_to_file(opm,spot_results)
multiplot_spot_to_file(opm,spot_results)

ray_abr_results = ray_abr_analysis(opm,num_rays=21,apply_vignetting=True)
plot_ray_abr_to_file(opm,ray_abr_results)

opt_results = opd_analysis(opm,num_rays=21,apply_vignetting=True)

print("done")