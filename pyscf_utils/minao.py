# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
from pyscf import gto
import numpy as np

def minao(mol):
    from pyscf.scf import atom_hf
    from pyscf.scf import addons
    import time

    times = []
    times.append(time.time())

    def minao_basis(symb, nelec_ecp):
        occ = []
        basis_ano = []
        if gto.is_ghost_atom(symb):
            return occ, basis_ano

        stdsymb = gto.mole._std_symbol(symb)
        basis_add = gto.basis.load('ano', stdsymb)
# coreshl defines the core shells to be removed in the initial guess
        coreshl = gto.ecp.core_configuration(nelec_ecp)
        #coreshl = (0,0,0,0)  # it keeps all core electrons in the initial guess
        for l in range(4):
            ndocc, frac = atom_hf.frac_occ(stdsymb, l)
            assert ndocc >= coreshl[l]
            degen = l * 2 + 1
            occ_l = [2,]*(ndocc-coreshl[l]) + [frac,]
            occ.append(np.repeat(occ_l, degen))
            basis_ano.append([l] + [b[:1] + b[1+coreshl[l]:ndocc+2]
                                    for b in basis_add[l][1:]])
        occ = np.hstack(occ)

        if nelec_ecp > 0:
            if symb in mol._basis:
                input_basis = mol._basis[symb]
            elif stdsymb in mol._basis:
                input_basis = mol._basis[stdsymb]
            else:
                raise KeyError(symb)

            basis4ecp = [[] for i in range(4)]
            for bas in input_basis:
                l = bas[0]
                if l < 4:
                    basis4ecp[l].append(bas)

            occ4ecp = []
            for l in range(4):
                nbas_l = sum((len(bas[1]) - 1) for bas in basis4ecp[l])
                ndocc, frac = atom_hf.frac_occ(stdsymb, l)
                ndocc -= coreshl[l]
                assert ndocc <= nbas_l

                occ_l = np.zeros(nbas_l)
                occ_l[:ndocc] = 2
                if frac > 0:
                    occ_l[ndocc] = frac
                occ4ecp.append(np.repeat(occ_l, l * 2 + 1))

            occ4ecp = np.hstack(occ4ecp)
            basis4ecp = lib.flatten(basis4ecp)

            atm1 = gto.Mole()
            atm2 = gto.Mole()
            atom = [[symb, (0.,0.,0.)]]
            atm1._atm, atm1._bas, atm1._env = atm1.make_env(atom, {symb:basis4ecp}, [])
            atm2._atm, atm2._bas, atm2._env = atm2.make_env(atom, {symb:basis_ano}, [])
            atm1._built = True
            atm2._built = True
            s12 = gto.intor_cross('int1e_ovlp', atm1, atm2)
            if abs(np.linalg.det(s12[occ4ecp>0][:,occ>0])) > .1:
                occ, basis_ano = occ4ecp, basis4ecp
            else:
                logger.debug(mol, 'Density of valence part of ANO basis '
                             'will be used as initial guess for %s', symb)
        return occ, basis_ano

    # Issue 548
    if any(gto.charge(mol.atom_symbol(ia)) > 96 for ia in range(mol.natm)):
        logger.info(mol, 'MINAO initial guess is not available for super-heavy '
                    'elements. "atom" initial guess is used.')
        return init_guess_by_atom(mol)


    times.append(time.time())
    nelec_ecp_dic = dict([(mol.atom_symbol(ia), mol.atom_nelec_core(ia))
                          for ia in range(mol.natm)])
    times.append(time.time())

    basis = {}
    occdic = {}
    for symb, nelec_ecp in nelec_ecp_dic.items():
        occ_add, basis_add = minao_basis(symb, nelec_ecp)
        occdic[symb] = occ_add
        basis[symb] = basis_add

    times.append(time.time())

    occ = []
    new_atom = []
    for ia in range(mol.natm):
        #print(ia)
        symb = mol.atom_symbol(ia)
        if not gto.is_ghost_atom(symb):
            occ.append(occdic[symb])
            new_atom.append(mol._atom[ia])
    occ = np.hstack(occ)

    times.append(time.time())

    pmol = gto.Mole()
    times.append(time.time())
    pmol._atm, pmol._bas, pmol._env = pmol.make_env(new_atom, basis, [])
    times.append(time.time())
    pmol._built = True
    dm = addons.project_dm_nr2nr(pmol, np.diag(occ), mol)

    times.append(time.time())

    times = np.array(times)
    # print("MINAO timing:", np.around(times[1:]-times[:-1], 2))

    return dm