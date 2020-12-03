import numpy as np
import inspect
import copy
import time

from pyscf.__all__ import *

# Functions in this dict have caching enabled
to_cache = {
    pbc.gto.Cell: ['energy_nuc', 'ewald'],
    pbc.scf.KRHF: ['get_hcore'],
    pbc.df.AFTDF: ['ft_loop'],
    pbc.df.MDF:   ['ft_loop'],
}

# Functions in this set will deep copy the cached result
copy_policy = {
}

# Arguments to ignore in each function when checking for a cached result
ignore_arguments = {
    pbc.df.AFTDF.ft_loop: ['max_memory'],
    pbc.df.MDF.ft_loop:   ['max_memory'],
}

# Try to check if two arbitrary variables are equal, not ideal
def same_val(a, b):
    if id(a) == id(b):
        return True
    elif isinstance(a, (str, int)):
        return a == b
    elif isinstance(a, lib.StreamObject):
        return False # since ids aren't the same
    else:
        if np.shape(a) != np.shape(b):
            return False
        return np.allclose(a, b, rtol=0, atol=1e-8)

# Requires that all args in cached function are actually kwargs
def cache(function):
    config = []
    result = []

    if function in copy_policy:
        do_copy = lambda x: copy.deepcopy(x)
    else:
        do_copy = lambda x: x

    argspec = inspect.getfullargspec(function)
    defaults = dict(zip(*[reversed(l) for l in (argspec.args[1:], argspec.defaults or [])]))
    is_generator = inspect.isgeneratorfunction(function)

    def wrapper(*args, **kwargs):
        cfg = defaults.copy()
        cfg.update({argspec.args[i]: args[i] for i in range(len(args))})
        cfg.update(kwargs)

        for i in range(len(config)):
            keys = set().union(config[i].keys(), cfg.keys())
            for key in ignore_arguments.get(function, []):
                keys.discard(key)

            for key in keys:
                a = config[i][key]
                b = cfg[key]
                if not same_val(a, b):
                    break
            else:
                return do_copy(result[i])

        res = function(**cfg)

        if is_generator:
            res = list(res)

        config.append(cfg)
        result.append(res)

        return res

    return wrapper

# Apply the cache decorator to the desired functions
def apply_cache():
    for cls, values in to_cache.items():
        class Tmp(cls):
            pass

        for value in values:
            func = getattr(Tmp, value)
            setattr(Tmp, value, cache(func))

        globals()[cls.__name__] = Tmp



if __name__ == '__main__':
    import warnings
    warnings.simplefilter('ignore')

    apply_cache()

    cell_data = dict(
        atom='He 1 0 1; He 0 0 1', 
        basis='6-31g', 
        a=np.eye(3)*3, 
        mesh=[6,]*3,
        verbose=0,
    )

    rhf_data = dict(
        df_type = 'aftdf',
        exxdiv='ewald',
        kpts=[1,1,2],
    )

    exp_to_discard = cell_data.pop('exp_to_discard', None)
    df_type = rhf_data.pop('df_type', 'mdf')
    kpts = rhf_data.pop('kpts', [1,1,1])

    t0 = time.time()

    cell_old = pbc.gto.C(**cell_data.copy())
    cell_old.exp_to_discard = exp_to_discard

    rhf_old = pbc.scf.KRHF(cell_old)
    for k, v in rhf_data.items():
        setattr(rhf_old, k, v)
    rhf_old.with_df = [pbc.df.FFTDF, pbc.df.AFTDF, pbc.df.GDF, pbc.df.MDF][['fftdf', 'aftdf', 'gdf', 'mdf'].index(df_type)](cell_old)
    rhf_old.kpts = cell_old.make_kpts(kpts)
    rhf_old.run()

    t1 = time.time()

    cell_new = Cell()
    cell_new.build(**cell_data.copy())
    cell_new.exp_to_discard = exp_to_discard

    rhf_new = KRHF(cell_new)
    for k, v in rhf_data.items():
        setattr(rhf_new, k, v)
    rhf_new.with_df = [pbc.df.FFTDF, AFTDF, pbc.df.GDF, MDF][['fftdf', 'aftdf', 'gdf', 'mdf'].index(df_type)](cell_old)
    rhf_new.kpts = cell_new.make_kpts(kpts)
    rhf_new.run()

    t2 = time.time()

    passed = True
    for key in ['e_tot', 'mo_energy', 'mo_occ', 'converged', 'make_rdm1']:
        a, b = getattr(rhf_old, key), getattr(rhf_new, key)
        if callable(a):
            a, b = a(), b()
        this_passed = np.allclose(a, b)
        if not this_passed:
            print(key, 'failed')
        passed = passed and this_passed

    print('%12.4f %12.4f %12s' % (t1-t0, t2-t1, passed))

