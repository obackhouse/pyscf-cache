import numpy as np
import time

from pyscf.pbc import gto, scf, df
import pyscf_cache


# Run without caching:

cell = gto.Cell()
cell.build(atom='He 0 0 1; He 1 0 1', basis='6-31g', a=np.eye(3)*3, mesh=[10,]*3, verbose=0)

rhf_old = scf.KRHF(cell)
rhf_old.with_df = df.AFTDF(cell)
rhf_old.kpts = cell.make_kpts([2,2,1])

t0 = time.time()
rhf_old.run()
t1 = time.time()


# Assign desired cached functions and apply decorators:

pyscf_cache.to_cache = {
    gto.Cell: ['energy_nuc', 'ewald'],
    scf.KRHF: ['get_hcore'],
    df.AFTDF: ['ft_loop'],
}

copy_policy = {}

ignore_arguments = {
    df.AFTDF.ft_loop: ['max_memory'],
}

pyscf_cache.apply_cache()


# Run with caching:

cell = pyscf_cache.Cell()
cell.build(atom='He 0 0 1; He 1 0 1', basis='6-31g', a=np.eye(3)*3, mesh=[10,]*3, verbose=0)

rhf_new = pyscf_cache.KRHF(cell)
rhf_new.with_df = pyscf_cache.AFTDF(cell)
rhf_new.kpts = cell.make_kpts([2,2,1])

t2 = time.time()
rhf_new.run()
t3 = time.time()


# Compare:

print('%16.12f   %.4f s' % (rhf_old.e_tot, t1-t0))
print('%16.12f   %.4f s' % (rhf_new.e_tot, t3-t2))
