from typing import List

from rayoptics.environment import *
from rayoptics.raytr.trace import setup_pupil_coords
from rayoptics.raytr.waveabr import wave_abr_full_calc

# Standalone versions of analysis

def eval_opd_fan(opt_model,fi,xy, num_rays=21, **kwargs):
    def opd(p, xy, ray_pkg, fld, wvl, foc):
        if ray_pkg[mc.ray] is not None:
            fod = opt_model['analysis_results']['parax_data'].fod
            opd = wave_abr_full_calc(fod, fld, wvl, foc, ray_pkg,
                                     fld.chief_ray, fld.ref_sphere)
            convert_to_waves = 1 / opt_model.nm_to_sys_units(wvl)
            return convert_to_waves * opd
        else:
            return None

    seq_model = opt_model.seq_model
    return seq_model.trace_fan(opd,fi,xy,num_rays=num_rays,**kwargs)

def opd_analysis(opm,num_rays=21,apply_vignetting=True):
    results = []
    fov = opm['osp']['fov']
    for fi,fld in enumerate(fov.fields):
        for xy in range(2):
            results.append(eval_opd_fan(opm,fi,xy,num_rays=num_rays,append_if_none=False,apply_vignetting=apply_vignetting))
    return results

def ray_abr(p, xy, ray_pkg, fld, wvl, foc):
    if ray_pkg[mc.ray] is not None:
        image_pt = fld.ref_sphere[0]
        ray = ray_pkg[mc.ray]
        dist = foc / ray[-1][mc.d][2]
        defocused_pt = ray[-1][mc.p] + dist * ray[-1][mc.d]
        t_abr = defocused_pt - image_pt
        return t_abr[xy]
    else:
        return None

def eval_abr_fan(opt_model, fi, xy, num_rays=21, **kwargs):
    seq_model = opt_model.seq_model
    return seq_model.trace_fan(ray_abr, fi, xy, num_rays=num_rays, **kwargs)

def ray_abr_analysis(opm,num_rays=21,apply_vignetting=True):
    results = []
    fov = opm['osp']['fov']
    for fi,fld in enumerate(fov.fields):
        for xy in range(2):
            results.append(eval_abr_fan(opm,fi,xy,num_rays=num_rays,append_if_none=False,apply_vignetting=apply_vignetting))
    return results

def generate_hexapolar_points(max_radius: float, num_rings: int)->List[np.ndarray]:
    points = []
    # center is taken as 0,0
    points.append(np.array([0.,0.]))
    step = max_radius / num_rings
    r = max_radius
    while r > 1e-8:
        astep = (step / r) * (math.pi / 3)
        a = 0
        while a < 2 * math.pi - 1e-8:
            points.append([math.sin(a) * r, math.cos(a) * r])
            a += astep
        r -= step
    return points

def trace_rings_wvl(opt_model, num_rings, fld, wvl, foc, img_filter=None,
               append_if_none=True, **kwargs):
    output_filter = kwargs.pop('output_filter', None)
    rayerr_filter = kwargs.pop('rayerr_filter', None)

    grid = []
    for pupil in generate_hexapolar_points(1.0,num_rings):
        ray_result = trace_safe(opt_model, pupil, fld, wvl,
                                output_filter, rayerr_filter,
                                check_apertures=True, **kwargs)

        if ray_result.pkg is not None:
            if img_filter:
                result = img_filter(pupil, ray_result.pkg)
                grid.append(result)
            else:
                grid.append([pupil[0], pupil[1], ray_result.pkg])
        else:  # ray outside pupil or failed
            if img_filter:
                result = img_filter(pupil, None)
                if result is not None or append_if_none:
                    grid.append(result)
            else:
                if append_if_none:
                    grid.append([pupil[0], pupil[1], None])

    return grid

def seq_model_trace_rings(opt_model, fct, fi, wl=None, num_rings=21,
               append_if_none=True, **kwargs):
    sm = opt_model.seq_model
    osp = sm.opt_model.optical_spec
    wvls = osp.spectral_region
    wvl = sm.central_wavelength()
    wv_list = wvls.wavelengths if wl is None else [wl]
    fld = osp.field_of_view.fields[fi]
    foc = osp.defocus.get_focus()

    rs_pkg, cr_pkg = setup_pupil_coords(sm.opt_model,
                                              fld, wvl, foc)
    fld.chief_ray = cr_pkg
    fld.ref_sphere = rs_pkg

    grids = []
    for wi, wvl in enumerate(wv_list):
        grid = trace_rings_wvl(sm.opt_model,num_rings, fld, wvl, foc,
                                append_if_none=append_if_none,
                                img_filter=lambda p, ray_pkg:
                                fct(p, wi, ray_pkg, fld, wvl, foc),
                                **kwargs)
        grids.append(grid)
    rc = wvls.render_colors
    return grids, rc

def spot(p, wi, ray_pkg, fld, wvl, foc):
    if ray_pkg is not None:
        image_pt = fld.ref_sphere[0]
        ray = ray_pkg[mc.ray]
        dist = foc / ray[-1][mc.d][2]
        defocused_pt = ray[-1][mc.p] + dist * ray[-1][mc.d]
        t_abr = defocused_pt - image_pt
        return np.array([t_abr[0], t_abr[1]])
    else:
        return None

def spot_analysis(opm,num_rings=21,apply_vignetting=True):
    results = []
    fov = opm['osp']['fov']
    for fi,fld in enumerate(fov.fields):
        results.append(seq_model_trace_rings(opm,spot,fi,wl=None,num_rings=num_rings,append_if_none=False,apply_vignetting=apply_vignetting))
    return results

def plot_spot_to_file(results):
    for fi,fld_result in enumerate(results):
        data, colors = fld_result
        plt.figure()
        for wi,wvl_result in enumerate(data):
            c = colors[wi]
            arr = np.stack(wvl_result)
            x = arr[:, 0]  # first column
            y = arr[:, 1]  # second column

            x *= 1000.
            y *= 1000.
            plt.scatter(x,y,c=c)

        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.title('Spot ' + str(fi))
        plt.grid(False)

        plt.savefig("spot" + str(fi) + ".svg", format="svg")
        plt.close()