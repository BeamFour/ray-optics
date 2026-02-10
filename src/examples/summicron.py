from rayoptics.environment import *
from rayoptics.raytr import vigcalc

opm = OpticalModel()
sm  = opm['seq_model']
osp = opm['optical_spec']
pm = opm['parax_model']
em = opm['ele_model']
pt = opm['part_tree']
ar = opm['analysis_results']
fov = osp['fov']
osp.pupil = PupilSpec(osp, key=['image', 'f/#'], value=2.0)
osp.field_of_view = FieldSpec(osp, key=('object', 'angle'), value=22.5, flds=[0.0,1.0], is_relative=True, is_wide_angle=True)
#osp.spectral_region = WvlSpec([(486.1327, 1.0), (587.5618, 1.0), (656.2725, 1.0)], ref_wl=1)
osp.spectral_region = WvlSpec([(587.5618, 1.0)], ref_wl=0)
opm.system_spec.title = "Leica Summicron R 50mm f/2"
opm.system_spec.dimensions = 'mm'
opm.radius_mode = True
sm.gaps[0].thi=1e10
sm.add_surface([42.71,3.99,'N-SF10','Schott'],sd=14.47)
sm.add_surface([195.38,0.2],sd=13.53)
sm.add_surface([20.5,7.18,'J-BASF6','Hikari'],sd=12.01)
sm.add_surface([0.0,1.29,'N-SF11','Schott'],sd=10.745)
sm.add_surface([14.94,5.35],sd=9.195)
sm.add_surface([0.0,7.61],sd=9.0295)
sm.set_stop()
sm.add_surface([-14.94,1.0,'N-SF2','Schott'],sd=8.75)
sm.add_surface([0.0,5.22,'N-LAF21','Schott'],sd=9.635)
sm.add_surface([-20.5,0.2],sd=10.19)
sm.add_surface([0.0,3.69,'N-LAF21','Schott'],sd=11.48)
sm.add_surface([-42.71,37.32],sd=11.985)
sm.do_apertures = False
opm.update_model()
sm.list_model()
#set_stop_aperture(opm)
set_pupil(opm)
#if fov.is_wide_angle:
#vigcalc.set_vig(opm,use_bisection=True)
#else:
#    trace.apply_paraxial_vignetting(opm)
sm.list_surfaces()
sm.list_gaps()
sm.list_model()
print('')
listobj(osp)

fov = osp['fov']
print(f"aim_info:  {fov.fields[0].aim_info}")
print(f"aim_info:  {fov.fields[1].aim_info}")

# List the optical specifications
pm.first_order_data()
# List the paraxial model
pm.list_lens()

#layout_plt = plt.figure(FigureClass=InteractiveLayout, opt_model=opm, do_draw_rays=True, do_paraxial_layout=False,
#                        is_dark=isdark).plot()
