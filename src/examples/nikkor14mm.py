from opticalglass import opticalmedium as om

from rayoptics import listobj
from rayoptics.elem.profiles import EvenPolynomial
from rayoptics.gui.appcmds import set_vignetting
from rayoptics.optical.opticalmodel import OpticalModel
from rayoptics.raytr.opticalspec import PupilSpec, FieldSpec, WvlSpec
from rayoptics.raytr import trace
from rayoptics.raytr.trace import list_ray
from rayoptics.raytr.wideangle import eval_real_image_ht

opm = OpticalModel()
sm  = opm.seq_model
osp = opm.optical_spec
pm = opm.parax_model
osp.pupil = PupilSpec(osp, key=['image', 'f/#'], value=4.0)
osp.field_of_view = FieldSpec(osp, key=['object', 'angle'], flds=[0.,57.68], is_wide_angle=True)
osp.spectral_region = WvlSpec([(486.1327, 0.5), (587.5618, 1.0), (656.2725, 0.5)], ref_wl=1)
opm.system_spec.title = "JP2019-008031 Example 1 (Nikon Nikkor Z 14-30mm f/4 S)"
opm.system_spec.dimensions = 'MM'
opm.radius_mode = True
sm.gaps[0].thi=1e10
sm.add_surface([190.7535,3.0,1.6937,53.32])
sm.ifcs[sm.cur_surface].max_aperture = 29.285
sm.add_surface([18.8098,9.5])
sm.ifcs[sm.cur_surface].profile = EvenPolynomial(r=18.8098, cc=-1.0,
	coefs=[0.0,-1.33157E-5,-3.07345E-8,6.9126E-11,-3.76684E-14,0.0,0.0])
sm.ifcs[sm.cur_surface].max_aperture = 22.485
sm.add_surface([51.563,2.9,1.6937,53.32])
sm.ifcs[sm.cur_surface].max_aperture = 19.205
sm.add_surface([22.702,9.7])
sm.ifcs[sm.cur_surface].profile = EvenPolynomial(r=22.702, cc=-1.0,
	coefs=[0.0,3.67009E-5,1.37031E-7,-5.20756E-10,3.14884E-12,-5.6153E-15,0.0])
sm.ifcs[sm.cur_surface].max_aperture = 14.475
sm.add_surface([-71.0651,1.9,1.49782,82.57])
sm.ifcs[sm.cur_surface].max_aperture = 15.05
sm.add_surface([44.4835,0.1])
sm.ifcs[sm.cur_surface].max_aperture = 15.05
sm.add_surface([32.608,4.5,1.90265,35.73])
sm.ifcs[sm.cur_surface].max_aperture = 15.05
sm.add_surface([296.5863,28.616])
sm.ifcs[sm.cur_surface].max_aperture = 15.05
sm.add_surface([63.0604,2.0,1.59349,67.0])
sm.ifcs[sm.cur_surface].max_aperture = 9.04
sm.add_surface([499.8755,0.1])
sm.ifcs[sm.cur_surface].max_aperture = 9.04
sm.add_surface([24.0057,1.2,1.883,40.66])
sm.ifcs[sm.cur_surface].max_aperture = 9.605
sm.add_surface([13.347,4.5,1.56883,56.0])
sm.ifcs[sm.cur_surface].max_aperture = 8.54
sm.add_surface([333.9818,2.5])
sm.ifcs[sm.cur_surface].max_aperture = 8.54
sm.add_surface([0.0,7.483])
sm.set_stop()
sm.ifcs[sm.cur_surface].max_aperture = 5.6335
sm.add_surface([36.3784,1.1,1.816,46.59])
sm.ifcs[sm.cur_surface].max_aperture = 8.59
sm.add_surface([14.0097,4.71,1.51612,64.08])
sm.ifcs[sm.cur_surface].max_aperture = 8.42
sm.add_surface([61.0448,0.2])
sm.ifcs[sm.cur_surface].profile = EvenPolynomial(r=61.0448, cc=0.0,
	coefs=[0.0,1.75905E-5,-6.64635E-8,2.26551E-10,-4.40763E-12,0.0,0.0])
sm.ifcs[sm.cur_surface].max_aperture = 8.42
sm.add_surface([27.9719,3.15,1.49782,82.57])
sm.ifcs[sm.cur_surface].max_aperture = 8.55
sm.add_surface([-75.3921,0.25])
sm.ifcs[sm.cur_surface].max_aperture = 8.55
sm.add_surface([91.9654,3.05,1.49782,82.57])
sm.ifcs[sm.cur_surface].max_aperture = 8.915
sm.add_surface([-29.3923,1.579])
sm.ifcs[sm.cur_surface].max_aperture = 8.915
sm.add_surface([72.093,1.0,1.795,45.31])
sm.ifcs[sm.cur_surface].max_aperture = 9.065
sm.add_surface([20.9929,5.766])
sm.ifcs[sm.cur_surface].max_aperture = 9.065
sm.add_surface([-538.2301,4.8,1.49782,82.57])
sm.ifcs[sm.cur_surface].max_aperture = 10.935
sm.add_surface([-20.1257,0.1])
sm.ifcs[sm.cur_surface].max_aperture = 10.935
sm.add_surface([-38.9341,1.4,1.76546,46.75])
sm.ifcs[sm.cur_surface].profile = EvenPolynomial(r=-38.9341, cc=-1.0,
	coefs=[0.0,-2.67902E-5,-3.34364E-8,-1.13765E-10,-1.88017E-13,0.0,0.0])
sm.ifcs[sm.cur_surface].max_aperture = 11.06
sm.add_surface([154.832,21.36])
sm.ifcs[sm.cur_surface].max_aperture = 11.815
sm.list_surfaces()
sm.list_gaps()
fov = osp['fov']
sm.do_apertures = False
opm.update_model()
set_vignetting(opm)
opm.update_model()
sm.list_model()
print("")
listobj(osp)

pm.first_order_data()

for fld in osp.field_of_view.fields:
    wvl = sm.central_wavelength()
    foc = osp.defocus.get_focus()
    rs, cr = trace.setup_pupil_coords(opm,fld,wvl,foc)
    list_ray(cr[0])
    print(f"op_delta:  {cr[0].op}")
    print(f"aim_info:  {fld.aim_info}")
    print(f"refsph r:  {rs[2]}")

result = eval_real_image_ht(opm,fov.fields[1],587.5618)
print(f"pt:  {result[0][0]}")
print(f"dir:  {result[0][1]}")
print(f"z_enp:  {result[1]}")

#all_fields = trace.trace_all_fields(opm)

#print(all_fields)

#rays = trace.trace_boundary_rays(opm)

print('done')