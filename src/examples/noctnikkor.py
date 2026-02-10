from rayoptics.environment import *

opm = OpticalModel()
sm  = opm['seq_model']
osp = opm['optical_spec']
pm = opm['parax_model']
em = opm['ele_model']
pt = opm['part_tree']
ar = opm['analysis_results']

osp.pupil = PupilSpec(osp, key=['image', 'f/#'], value=1.2)
osp['fov'].is_relative = True
osp['fov'].set_from_list([0.0, 0.1, 0.3, 0.5, 0.7, 1.0])
osp['fov'].is_wide_angle = True
osp['fov'].key = ('object', 'angle')
osp['fov'].value = 20.45
osp.spectral_region = WvlSpec([(486.1327, 0.5), (587.5618, 1.0), (656.2725, 0.5)], ref_wl=1)
opm.system_spec.title = "null"
opm.system_spec.dimensions = 'mm'
opm.radius_mode = True
sm.gaps[0].thi=1e10
sm.add_surface([79.9175025,6.885,1.795,45.31])
sm.ifcs[sm.cur_surface].profile = EvenPolynomial(r=79.9175025, cc=0.0,
	coefs=[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
sm.ifcs[sm.cur_surface].max_aperture = 25.24375
sm.add_surface([0.0,0.1])
sm.ifcs[sm.cur_surface].max_aperture = 25.24375
sm.add_surface([33.737,9.75,1.8485,43.79])
sm.ifcs[sm.cur_surface].max_aperture = 22.416
sm.add_surface([70.18675,1.56])
sm.ifcs[sm.cur_surface].max_aperture = 22.416
sm.add_surface([134.505,2.87,1.74,28.3])
sm.ifcs[sm.cur_surface].max_aperture = 21.0845
sm.add_surface([22.3687,8.44])
sm.ifcs[sm.cur_surface].max_aperture = 16.064205
sm.add_surface([0.0,7.95])
sm.set_stop()
sm.ifcs[sm.cur_surface].max_aperture = 15.6135
sm.add_surface([-23.02418,1.64,1.74077,27.79])
sm.ifcs[sm.cur_surface].max_aperture = 15.7225
sm.add_surface([306.553,8.196,1.788,47.37])
sm.ifcs[sm.cur_surface].max_aperture = 20.1
sm.add_surface([-37.555,0.15])
sm.ifcs[sm.cur_surface].max_aperture = 20.1
sm.add_surface([-396.94,6.147,1.7725,46.62])
sm.ifcs[sm.cur_surface].max_aperture = 19.75
sm.add_surface([-52.56789,0.0])
sm.ifcs[sm.cur_surface].max_aperture = 19.75
sm.add_surface([223.8426,4.016,1.795,45.31])
sm.ifcs[sm.cur_surface].max_aperture = 19.1375
sm.add_surface([-94.08052,37.78])
sm.ifcs[sm.cur_surface].max_aperture = 19.1375
sm.do_apertures = False
opm.update_model()
#set_vignetting(opm)
#apply_paraxial_vignetting(opm)
sm.list_model()
#set_stop_aperture(opm)
set_pupil(opm)
sm.list_model()
listobj(osp)

fov = osp['fov']
print(f"aim_info:  {fov.fields[0].aim_info}")
print(f"aim_info:  {fov.fields[1].aim_info}")
print(f"aim_info:  {fov.fields[2].aim_info}")
print(f"aim_info:  {fov.fields[3].aim_info}")
print(f"aim_info:  {fov.fields[4].aim_info}")
print(f"aim_info:  {fov.fields[5].aim_info}")