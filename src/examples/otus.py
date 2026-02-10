from rayoptics.environment import *

opm = OpticalModel()
sm  = opm['seq_model']
osp = opm['optical_spec']
pm = opm['parax_model']
em = opm['ele_model']
pt = opm['part_tree']
ar = opm['analysis_results']
osp.pupil = PupilSpec(osp, key=['image', 'f/#'], value=1.43857)
osp.field_of_view = FieldSpec(osp, key=('object', 'angle'), value=21.85, flds=[0.0,0.7,1.0], is_relative=True, is_wide_angle=False)
osp.spectral_region = WvlSpec([(486.1327, 0.5), (587.5618, 1.0), (656.2725, 0.5)], ref_wl=1)
opm.system_spec.title = "Zeiss Otus 55mm f1.4"
opm.system_spec.dimensions = 'mm'
opm.radius_mode = True
sm.gaps[0].thi=1e10
sm.add_surface([225.36466834503,2.1033312091354,'S-FSL5','Ohara'],sd=28.0)
sm.add_surface([45.736909912913,16.816476607346],sd=24.0)
sm.add_surface([-43.4839710998151,2.162275148848,'S-NBM51','Ohara'],sd=24.0)
sm.add_surface([160.511398683478,2.4078320111969],sd=28.0)
sm.add_surface([208.941209674516,11.386852109962,'S-PHM52','Ohara'],sd=28.0)
sm.add_surface([-54.306190349513,0.0],sd=28.0)
sm.add_surface([61.5685548331634,9.97359385999999,'S-FPL53','Ohara'],sd=28.5)
sm.add_surface([-272.694326590584,9.23642559062],sd=28.5)
sm.add_surface([112.954001341021,4.392370735944,'S-LAH99','Ohara'],sd=27.25)
sm.add_surface([3740.84709952516,0.0],sd=27.25)
sm.add_surface([33.9171015242564,14.4865453457114,'S-FPM2','Ohara'],sd=24.0)
sm.add_surface([-122.898138652455,2.2092009079593,'S-NBH5','Ohara'],sd=23.0)
sm.add_surface([23.78728047,8.31695997266627],sd=17.0)
sm.add_surface([0.0,2.992078158],sd=16.00795)
sm.set_stop()
sm.add_surface([-148.328915311997,1.434202797068,'S-NBH53V','Ohara'],sd=15.7)
sm.add_surface([24.1902218579498,6.98151570199999,1.874,35.26],sd=15.5)  # 'S-LAH75','Ohara'
sm.add_surface([374.552470798813,4.488117237],sd=15.5)
sm.add_surface([-28.3396604218354,1.37780112643031,'S-NBH52V','Ohara'],sd=14.7)
sm.add_surface([140.452377398,0.91757063512],sd=15.45905)
sm.add_surface([67.7398806672967,5.0262923616856,'S-FPL51','Ohara'],sd=16.0)
sm.add_surface([-54.6681021516486,0.31583379676462],sd=16.0)
sm.add_surface([127.953325269898,5.87378254218892,1.583126,59.38],sd=16.0)  # 'L-BAL42','Ohara'
sm.ifcs[sm.cur_surface].profile = EvenPolynomial(r=127.953325269898, cc=0.0,
	coefs=[0.0,-6.06936532178685E-6,4.5567704355776E-9,-3.19548147113261E-11,4.29902435086771E-14,-5.0E-18,0.0,0.0,0.0,0.0])
sm.add_surface([-33.15760489,36.66500040938],sd=16.0)
sm.ifcs[sm.cur_surface].profile = EvenPolynomial(r=-33.15760489, cc=0.0,
	coefs=[0.0,3.39691207913842E-6,1.4343503783496E-9,-1.27723154320818E-11,1.37648185236955E-14,-1.0E-18,0.0,0.0,0.0,0.0])
sm.add_surface([0.0,2.0,'S-BSL7','Ohara'],sd=22.0)
sm.list_surfaces()
sm.list_gaps()
sm.do_apertures = False
opm.update_model()
set_vignetting(opm)
print('')
listobj(osp)
sm.list_model()
# List the optical specifications
pm.first_order_data()
# List the paraxial model
pm.list_lens()
