import pyscf
import numpy as np 
from pyscf.gto import moleintor
import ctypes

mol = pyscf.gto.Mole(atom="C 0 0 0; C 0 0 1;", basis="sto3g")
mol.build()

class _cintoptHandler(ctypes.c_void_p):
    def __del__(self):
        try:
            if self.intor[:3] == 'ECP':
                libcgto.ECPdel_optimizer(ctypes.byref(self))
            else:
                libcgto.CINTdel_optimizer(ctypes.byref(self))
        except AttributeError:
            pass
def make_cintopt(atm, bas, env, intor):
    intor = intor.replace('_sph','').replace('_cart','').replace('_spinor','')
    c_atm = numpy.asarray(atm, dtype=numpy.int32, order='C')
    c_bas = numpy.asarray(bas, dtype=numpy.int32, order='C')
    c_env = numpy.asarray(env, dtype=numpy.double, order='C')
    natm = c_atm.shape[0]
    nbas = c_bas.shape[0]
    cintopt = ctypes.cast(lib.c_null_ptr(), _cintoptHandler)
    cintopt.intor = intor

    # TODO: call specific ECP optimizers for each intor.
    if intor[:3] == 'ECP':
        AS_ECPBAS_OFFSET = 18  # from gto/mole.py
        if env[AS_ECPBAS_OFFSET] == 0:
            raise RuntimeError('ecpbas or env is not properly initialized')
        foptinit = libcgto.ECPscalar_optimizer
    else:
        foptinit = getattr(libcgto, intor+'_optimizer')
    foptinit(ctypes.byref(cintopt),
             c_atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(natm),
             c_bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(nbas),
             c_env.ctypes.data_as(ctypes.c_void_p))
    return cintopt


def make_loc(bas, key):
    if 'cart' in key:
        l = bas[:,ANG_OF]
        dims = (l+1)*(l+2)//2 * bas[:,NCTR_OF]
    elif 'sph' in key:
        dims = (bas[:,ANG_OF]*2+1) * bas[:,NCTR_OF]
    else:  # spinor
        l = bas[:,ANG_OF]
        k = bas[:,KAPPA_OF]
        dims = (l*4+2) * bas[:,NCTR_OF]
        dims[k<0] = (l[k<0] * 2 + 2) * bas[k<0,NCTR_OF]
        dims[k>0] = (l[k>0] * 2    ) * bas[k>0,NCTR_OF]

    ao_loc = numpy.empty(len(dims)+1, dtype=numpy.int32)
    ao_loc[0] = 0
    dims.cumsum(dtype=numpy.int32, out=ao_loc[1:])
    return ao_loc


def getints4c(intor_name, atm, bas, env, shls_slice=None, comp=1,
              aosym='s1', ao_loc=None, cintopt=None, out=None):
    aosym = _stand_sym_code(aosym)
    atm = numpy.asarray(atm, dtype=numpy.int32, order='C')
    bas = numpy.asarray(bas, dtype=numpy.int32, order='C')
    env = numpy.asarray(env, dtype=numpy.double, order='C')
    c_atm = atm.ctypes.data_as(ctypes.c_void_p)
    c_bas = bas.ctypes.data_as(ctypes.c_void_p)
    c_env = env.ctypes.data_as(ctypes.c_void_p)
    natm = atm.shape[0]
    nbas = bas.shape[0]
    ao_loc = make_loc(bas, intor_name)

    if '_spinor' in intor_name:
        assert (aosym == 's1')

    if aosym == 's8':
        assert (shls_slice is None)
        from pyscf.scf import _vhf
        nao = ao_loc[-1]
        nao_pair = nao*(nao+1)//2
        out = numpy.ndarray((nao_pair*(nao_pair+1)//2), buffer=out)
        if nao_pair == 0:
            return out

        if cintopt is None:
            cintopt = make_cintopt(atm, bas, env, intor_name)
        drv = _vhf.libcvhf.GTO2e_cart_or_sph
        drv(getattr(libcgto, intor_name), cintopt,
            out.ctypes.data_as(ctypes.c_void_p),
            ao_loc.ctypes.data_as(ctypes.c_void_p),
            c_atm, ctypes.c_int(natm), c_bas, ctypes.c_int(nbas), c_env)
        return out

    else:
        if shls_slice is None:
            shls_slice = (0, nbas, 0, nbas, 0, nbas, 0, nbas)
        elif len(shls_slice) == 4:
            shls_slice = shls_slice + (0, nbas, 0, nbas)
        else:
            assert (shls_slice[1] <= nbas and shls_slice[3] <= nbas and
                   shls_slice[5] <= nbas and shls_slice[7] <= nbas)
        i0, i1, j0, j1, k0, k1, l0, l1 = shls_slice
        naoi = ao_loc[i1] - ao_loc[i0]
        naoj = ao_loc[j1] - ao_loc[j0]
        naok = ao_loc[k1] - ao_loc[k0]
        naol = ao_loc[l1] - ao_loc[l0]
        if aosym in ('s4', 's2ij'):
            nij = [naoi * (naoi + 1) // 2]
            assert (numpy.all(ao_loc[i0:i1]-ao_loc[i0] == ao_loc[j0:j1]-ao_loc[j0]))
        else:
            nij = [naoi, naoj]
        if aosym in ('s4', 's2kl'):
            nkl = [naok * (naok + 1) // 2]
            assert (numpy.all(ao_loc[k0:k1]-ao_loc[k0] == ao_loc[l0:l1]-ao_loc[l0]))
        else:
            nkl = [naok, naol]
        shape = [comp] + nij + nkl

        fill = getattr(libcgto, 'GTOnr2e_fill_'+aosym)
        out = numpy.ndarray(shape, buffer=out)

        if out.size > 0:
            if cintopt is None:
                cintopt = make_cintopt(atm, bas, env, intor_name)
            prescreen = lib.c_null_ptr()
            print(intor_name)
            libcgto.GTOnr2e_fill_drv(libcgto.int2e_ip1_sph, libcgto.GTOnr2e_fill_s1, prescreen,
                out.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(comp),
                (ctypes.c_int*8)(*shls_slice),
                ao_loc.ctypes.data_as(ctypes.c_void_p), cintopt,
                c_atm, ctypes.c_int(natm), c_bas, ctypes.c_int(nbas), c_env)

        if comp == 1:
            out = out[0]
        return out

import warnings
import ctypes
import numpy
from pyscf import lib

libcgto = lib.load_library('libcgto')

ANG_OF     = 1
NPRIM_OF   = 2
NCTR_OF    = 3
KAPPA_OF   = 4
PTR_EXP    = 5
PTR_COEFF  = 6
BAS_SLOTS  = 8
NGRIDS     = 11
PTR_GRIDS  = 12



def getints(intor_name, atm, bas, env, shls_slice=None, comp=None, hermi=0,
            aosym='s1', ao_loc=None, cintopt=None, out=None):
    r'''1e and 2e integral generator.

    Args:
        intor_name : str

            ================================  =============
            Function                          Expression
            ================================  =============
            "int1e_ovlp"                      ( \| \)
            "int1e_nuc"                       ( \| nuc \| \)
            "int1e_kin"                       (.5 \| p dot p\)
            "int1e_ia01p"                     (#C(0 1) \| nabla-rinv \| cross p\)
            "int1e_giao_irjxp"                (#C(0 1) \| r cross p\)
            "int1e_cg_irxp"                   (#C(0 1) \| rc cross p\)
            "int1e_giao_a11part"              (-.5 \| nabla-rinv \| r\)
            "int1e_cg_a11part"                (-.5 \| nabla-rinv \| rc\)
            "int1e_a01gp"                     (g \| nabla-rinv cross p \|\)
            "int1e_igkin"                     (#C(0 .5) g \| p dot p\)
            "int1e_igovlp"                    (#C(0 1) g \|\)
            "int1e_ignuc"                     (#C(0 1) g \| nuc \|\)
            "int1e_z"                         ( \| zc \| \)
            "int1e_zz"                        ( \| zc zc \| \)
            "int1e_r"                         ( \| rc \| \)
            "int1e_r2"                        ( \| rc dot rc \| \)
            "int1e_rr"                        ( \| rc rc \| \)
            "int1e_rrr"                       ( \| rc rc rc \| \)
            "int1e_rrrr"                      ( \| rc rc rc rc \| \)
            "int1e_pnucp"                     (p* \| nuc dot p \| \)
            "int1e_prinvxp"                   (p* \| rinv cross p \| \)
            "int1e_ipovlp"                    (nabla \|\)
            "int1e_ipkin"                     (.5 nabla \| p dot p\)
            "int1e_ipnuc"                     (nabla \| nuc \|\)
            "int1e_iprinv"                    (nabla \| rinv \|\)
            "int1e_rinv"                      (\| rinv \|\)
            "int1e_pnucxp"                    (p* \| nuc cross p \| \)
            "int1e_irp"                       ( \| rc nabla \| \)
            "int1e_irrp"                      ( \| rc rc nabla \| \)
            "int1e_irpr"                      ( \| rc nabla rc \| \)
            "int1e_ggovlp"                    ( \| g g \| \)
            "int1e_ggkin"                     (.5 \| g g p dot p \| \)
            "int1e_ggnuc"                     ( \| g g nuc \| \)
            "int1e_grjxp"                     ( \| g r cross p \| \)
            "ECPscalar"                       AREP ECP integrals, similar to int1e_nuc
            "ECPscalar_ipnuc"                 (nabla i | ECP | ), similar to int1e_ipnuc
            "ECPscalar_iprinv"                similar to int1e_iprinv for a specific atom
            "ECPscalar_ignuc"                 similar to int1e_ignuc
            "ECPscalar_iprinvip"              similar to int1e_iprinvip
            "ECPso"                           < | Spin-orbit ECP | >
            "int1e_ovlp_spinor"               ( \| \)
            "int1e_nuc_spinor"                ( \| nuc \|\)
            "int1e_srsr_spinor"               (sigma dot r \| sigma dot r\)
            "int1e_sr_spinor"                 (sigma dot r \|\)
            "int1e_srsp_spinor"               (sigma dot r \| sigma dot p\)
            "int1e_spsp_spinor"               (sigma dot p \| sigma dot p\)
            "int1e_sp_spinor"                 (sigma dot p \|\)
            "int1e_spnucsp_spinor"            (sigma dot p \| nuc \| sigma dot p\)
            "int1e_srnucsr_spinor"            (sigma dot r \| nuc \| sigma dot r\)
            "int1e_govlp_spinor"              (g \|\)
            "int1e_gnuc_spinor"               (g \| nuc \|\)
            "int1e_cg_sa10sa01_spinor"        (.5 sigma cross rc \| sigma cross nabla-rinv \|\)
            "int1e_cg_sa10sp_spinor"          (.5 rc cross sigma \| sigma dot p\)
            "int1e_cg_sa10nucsp_spinor"       (.5 rc cross sigma \| nuc \| sigma dot p\)
            "int1e_giao_sa10sa01_spinor"      (.5 sigma cross r \| sigma cross nabla-rinv \|\)
            "int1e_giao_sa10sp_spinor"        (.5 r cross sigma \| sigma dot p\)
            "int1e_giao_sa10nucsp_spinor"     (.5 r cross sigma \| nuc \| sigma dot p\)
            "int1e_sa01sp_spinor"             (\| nabla-rinv cross sigma \| sigma dot p\)
            "int1e_spgsp_spinor"              (g sigma dot p \| sigma dot p\)
            "int1e_spgnucsp_spinor"           (g sigma dot p \| nuc \| sigma dot p\)
            "int1e_spgsa01_spinor"            (g sigma dot p \| nabla-rinv cross sigma \|\)
            "int1e_spspsp_spinor"             (sigma dot p \| sigma dot p sigma dot p\)
            "int1e_spnuc_spinor"              (sigma dot p \| nuc \|\)
            "int1e_ipovlp_spinor"             (nabla \|\)
            "int1e_ipkin_spinor"              (.5 nabla \| p dot p\)
            "int1e_ipnuc_spinor"              (nabla \| nuc \|\)
            "int1e_iprinv_spinor"             (nabla \| rinv \|\)
            "int1e_ipspnucsp_spinor"          (nabla sigma dot p \| nuc \| sigma dot p\)
            "int1e_ipsprinvsp_spinor"         (nabla sigma dot p \| rinv \| sigma dot p\)
            "int1e_grids"                     ( \| 1/r_grids \| \)
            "int1e_grids_spinor"              ( \| 1/r_grids \| \)
            "int1e_grids_ip"                  (nabla \| 1/r_grids \| \)
            "int1e_grids_ip_spinor"           (nabla \| 1/r_grids \| \)
            "int1e_grids_spvsp_spinor"        (sigma dot p \| 1/r_grids \| sigma dot p\)
            "int2e"                           ( \, \| \, \)
            "int2e_ig1"                       (#C(0 1) g \, \| \, \)
            "int2e_gg1"                       (g g \, \| \, \)
            "int2e_g1g2"                      (g \, \| g \, \)
            "int2e_p1vxp1"                    ( p* \, cross p \| \, \) ; SSO
            "int2e_spinor"                    (, \| \, \)
            "int2e_spsp1_spinor"              (sigma dot p \, sigma dot p \| \, \)
            "int2e_spsp1spsp2_spinor"         (sigma dot p \, sigma dot p \| sigma dot p \, sigma dot p \)
            "int2e_srsr1_spinor"              (sigma dot r \, sigma dot r \| \,\)
            "int2e_srsr1srsr2_spinor"         (sigma dot r \, sigma dot r \| sigma dot r \, sigma dot r\)
            "int2e_cg_sa10sp1_spinor"         (.5 rc cross sigma \, sigma dot p \| \,\)
            "int2e_cg_sa10sp1spsp2_spinor"    (.5 rc cross sigma \, sigma dot p \| sigma dot p \, sigma dot p \)
            "int2e_giao_sa10sp1_spinor"       (.5 r cross sigma \, sigma dot p \| \,\)
            "int2e_giao_sa10sp1spsp2_spinor"  (.5 r cross sigma \, sigma dot p \| sigma dot p \, sigma dot p \)
            "int2e_g1_spinor"                 (g \, \| \,\)
            "int2e_spgsp1_spinor"             (g sigma dot p \, sigma dot p \| \,\)
            "int2e_g1spsp2_spinor"            (g \, \| sigma dot p \, sigma dot p\)
            "int2e_spgsp1spsp2_spinor"        (g sigma dot p \, sigma dot p \| sigma dot p \, sigma dot p\)
            "int2e_spv1_spinor"               (sigma dot p \, \| \,\)
            "int2e_vsp1_spinor"               (\, sigma dot p \| \,\)
            "int2e_spsp2_spinor"              (\, \| sigma dot p \, sigma dot p\)
            "int2e_spv1spv2_spinor"           (sigma dot p \, \| sigma dot p \,\)
            "int2e_vsp1spv2_spinor"           (\, sigma dot p \| sigma dot p \,\)
            "int2e_spv1vsp2_spinor"           (sigma dot p \, \| \, sigma dot p\)
            "int2e_vsp1vsp2_spinor"           (\, sigma dot p \| \, sigma dot p\)
            "int2e_spv1spsp2_spinor"          (sigma dot p \, \| sigma dot p \, sigma dot p\)
            "int2e_vsp1spsp2_spinor"          (\, sigma dot p \| sigma dot p \, sigma dot p\)
            "int2e_ig1"                       (#C(0 1) g \, \| \, \)
            "int2e_ip1"                       (nabla \, \| \,\)
            "int2e_ip1_spinor"                (nabla \, \| \,\)
            "int2e_ipspsp1_spinor"            (nabla sigma dot p \, sigma dot p \| \,\)
            "int2e_ip1spsp2_spinor"           (nabla \, \| sigma dot p \, sigma dot p\)
            "int2e_ipspsp1spsp2_spinor"       (nabla sigma dot p \, sigma dot p \| sigma dot p \, sigma dot p\)
            "int2e_ipsrsr1_spinor"            (nabla sigma dot r \, sigma dot r \| \,\)
            "int2e_ip1srsr2_spinor"           (nabla \, \| sigma dot r \, sigma dot r\)
            "int2e_ipsrsr1srsr2_spinor"       (nabla sigma dot r \, sigma dot r \| sigma dot r \, sigma dot r\)
            "int2e_ip1"                       (nabla \, \| \,\)
            "int2e_ssp1ssp2_spinor"           ( \, sigma dot p \| gaunt \| \, sigma dot p\)
            "int2e_cg_ssa10ssp2_spinor"       (rc cross sigma \, \| gaunt \| \, sigma dot p\)
            "int2e_giao_ssa10ssp2_spinor"     (r cross sigma  \, \| gaunt \| \, sigma dot p\)
            "int2e_gssp1ssp2_spinor"          (g \, sigma dot p  \| gaunt \| \, sigma dot p\)
            "int2e_ipip1"                     ( nabla nabla \, \| \, \)
            "int2e_ipvip1"                    ( nabla \, nabla \| \, \)
            "int2e_ip1ip2"                    ( nabla \, \| nabla \, \)
            "int3c2e_ip1"                     (nabla \, \| \)
            "int3c2e_ip2"                     ( \, \| nabla\)
            "int2c2e_ip1"                     (nabla \| r12 \| \)
            "int3c2e_spinor"                  (nabla \, \| \)
            "int3c2e_spsp1_spinor"            (nabla \, \| \)
            "int3c2e_ip1_spinor"              (nabla \, \| \)
            "int3c2e_ip2_spinor"              ( \, \| nabla\)
            "int3c2e_ipspsp1_spinor"          (nabla sigma dot p \, sigma dot p \| \)
            "int3c2e_spsp1ip2_spinor"         (sigma dot p \, sigma dot p \| nabla \)
            "ECPscalar_spinor"                AREP ECP integrals, similar to int1e_nuc
            "ECPscalar_ipnuc_spinor"          (nabla i | ECP | ), similar to int1e_ipnuc
            "ECPscalar_iprinv_spinor"         similar to int1e_iprinv for a specific atom
            "ECPscalar_ignuc_spinor"          similar to int1e_ignuc
            "ECPscalar_iprinvip_spinor"       similar to int1e_iprinvip
            "ECPso_spinor"                    < | sigam dot Spin-orbit ECP | >
            ================================  =============

        atm : int32 ndarray
            libcint integral function argument
        bas : int32 ndarray
            libcint integral function argument
        env : float64 ndarray
            libcint integral function argument

    Kwargs:
        shls_slice : 8-element list
            (ish_start, ish_end, jsh_start, jsh_end, ksh_start, ksh_end, lsh_start, lsh_end)
        comp : int
            Components of the integrals, e.g. int1e_ipovlp has 3 components.
        hermi : int (1e integral only)
            Symmetry of the 1e integrals

            | 0 : no symmetry assumed (default)
            | 1 : hermitian
            | 2 : anti-hermitian

        aosym : str (2e integral only)
            Symmetry of the 2e integrals

            | 4 or '4' or 's4': 4-fold symmetry (default)
            | '2ij' or 's2ij' : symmetry between i, j in (ij|kl)
            | '2kl' or 's2kl' : symmetry between k, l in (ij|kl)
            | 1 or '1' or 's1': no symmetry

        out : ndarray (2e integral only)
            array to store the 2e AO integrals

    Returns:
        ndarray of 1-electron integrals, can be either 2-dim or 3-dim, depending on comp

    Examples:

    >>> mol.build(atom='H 0 0 0; H 0 0 1.1', basis='sto-3g')
    >>> gto.getints('int1e_ipnuc_sph', mol._atm, mol._bas, mol._env, comp=3) # <nabla i | V_nuc | j>
    [[[ 0.          0.        ]
      [ 0.          0.        ]]
     [[ 0.          0.        ]
      [ 0.          0.        ]]
     [[ 0.10289944  0.48176097]
      [-0.48176097 -0.10289944]]]
    '''
    intor_name, comp = _get_intor_and_comp(intor_name, comp)
    if any(bas[:,ANG_OF] > 12):
        raise NotImplementedError('cint library does not support high angular (l>12) GTOs')

    return getints4c(intor_name, atm, bas, env, shls_slice, comp,
                         aosym, ao_loc, cintopt, out)


_INTOR_FUNCTIONS = {
    # Functiona name            : (comp-for-scalar, comp-for-spinor)
    'int1e_ovlp'                : (1, 1),
    'int1e_nuc'                 : (1, 1),
    'int1e_kin'                 : (1, 1),
    'int1e_ia01p'               : (3, 3),
    'int1e_giao_irjxp'          : (3, 3),
    'int1e_cg_irxp'             : (3, 3),
    'int1e_giao_a11part'        : (9, 9),
    'int1e_cg_a11part'          : (9, 9),
    'int1e_a01gp'               : (9, 9),
    'int1e_igkin'               : (3, 3),
    'int1e_igovlp'              : (3, 3),
    'int1e_ignuc'               : (3, 3),
    'int1e_pnucp'               : (1, 1),
    'int1e_z'                   : (1, 1),
    'int1e_zz'                  : (1, 1),
    'int1e_r'                   : (3, 3),
    'int1e_r2'                  : (1, 1),
    'int1e_r4'                  : (1, 1),
    'int1e_rr'                  : (9, 9),
    'int1e_rrr'                 : (27, 27),
    'int1e_rrrr'                : (81, 81),
    'int1e_z_origj'             : (1, 1),
    'int1e_zz_origj'            : (1, 1),
    'int1e_r_origj'             : (3, 3),
    'int1e_rr_origj'            : (9, 9),
    'int1e_r2_origj'            : (1, 1),
    'int1e_r4_origj'            : (1, 1),
    'int1e_p4'                  : (1, 1),
    'int1e_prinvp'              : (1, 1),
    'int1e_prinvxp'             : (3, 3),
    'int1e_pnucxp'              : (3, 3),
    'int1e_irp'                 : (9, 9),
    'int1e_irrp'                : (27, 27),
    'int1e_irpr'                : (27, 27),
    'int1e_ggovlp'              : (9, 9),
    'int1e_ggkin'               : (9, 9),
    'int1e_ggnuc'               : (9, 9),
    'int1e_grjxp'               : (9, 9),
    'int2e'                     : (1, 1),
    'int2e_ig1'                 : (3, 3),
    'int2e_gg1'                 : (9, 9),
    'int2e_g1g2'                : (9, 9),
    'int2e_ip1v_rc1'            : (9, 9),
    'int2e_ip1v_r1'             : (9, 9),
    'int2e_ipvg1_xp1'           : (9, 9),
    'int2e_ipvg2_xp1'           : (9, 9),
    'int2e_p1vxp1'              : (3, 3),
    'int1e_inuc_rcxp'           : (3, 3),
    'int1e_inuc_rxp'            : (3, 3),
    'int1e_sigma'               : (12,3),
    'int1e_spsigmasp'           : (12,3),
    'int1e_srsr'                : (4, 1),
    'int1e_sr'                  : (4, 1),
    'int1e_srsp'                : (4, 1),
    'int1e_spsp'                : (4, 1),
    'int1e_sp'                  : (4, 1),
    'int1e_spnucsp'             : (4, 1),
    'int1e_sprinvsp'            : (4, 1),
    'int1e_srnucsr'             : (4, 1),
    'int1e_sprsp'               : (12,3),
    'int1e_govlp'               : (3, 3),
    'int1e_gnuc'                : (3, 3),
    'int1e_cg_sa10sa01'         : (36,9),
    'int1e_cg_sa10sp'           : (12,3),
    'int1e_cg_sa10nucsp'        : (12,3),
    'int1e_giao_sa10sa01'       : (36,9),
    'int1e_giao_sa10sp'         : (12,3),
    'int1e_giao_sa10nucsp'      : (12,3),
    'int1e_sa01sp'              : (12,3),
    'int1e_spgsp'               : (12,3),
    'int1e_spgnucsp'            : (12,3),
    'int1e_spgsa01'             : (36,9),
    'int2e_spsp1'               : (4, 1),
    'int2e_spsp1spsp2'          : (16,1),
    'int2e_srsr1'               : (4, 1),
    'int2e_srsr1srsr2'          : (16,1),
    'int2e_cg_sa10sp1'          : (12,3),
    'int2e_cg_sa10sp1spsp2'     : (48,3),
    'int2e_giao_sa10sp1'        : (12,3),
    'int2e_giao_sa10sp1spsp2'   : (48,3),
    'int2e_g1'                  : (12,3),
    'int2e_spgsp1'              : (12,3),
    'int2e_g1spsp2'             : (12,3),
    'int2e_spgsp1spsp2'         : (48,3),
    'int2e_pp1'                 : (1, 1),
    'int2e_pp2'                 : (1, 1),
    'int2e_pp1pp2'              : (1, 1),
    'int1e_spspsp'              : (4, 1),
    'int1e_spnuc'               : (4, 1),
    'int2e_spv1'                : (4, 1),
    'int2e_vsp1'                : (4, 1),
    'int2e_spsp2'               : (4, 1),
    'int2e_spv1spv2'            : (16,1),
    'int2e_vsp1spv2'            : (16,1),
    'int2e_spv1vsp2'            : (16,1),
    'int2e_vsp1vsp2'            : (16,1),
    'int2e_spv1spsp2'           : (16,1),
    'int2e_vsp1spsp2'           : (16,1),
    'int1e_ipovlp'              : (3, 3),
    'int1e_ipkin'               : (3, 3),
    'int1e_ipnuc'               : (3, 3),
    'int1e_iprinv'              : (3, 3),
    'int1e_rinv'                : (1, 1),
    'int1e_ipspnucsp'           : (12,3),
    'int1e_ipsprinvsp'          : (12,3),
    'int1e_ippnucp'             : (3, 3),
    'int1e_ipprinvp'            : (3, 3),
    'int2e_ip1'                 : (3, 3),
    'int2e_ip2'                 : (3, 3),
    'int2e_ipspsp1'             : (12,3),
    'int2e_ip1spsp2'            : (12,3),
    'int2e_ipspsp1spsp2'        : (48,3),
    'int2e_ipsrsr1'             : (12,3),
    'int2e_ip1srsr2'            : (12,3),
    'int2e_ipsrsr1srsr2'        : (48,3),
    'int2e_ssp1ssp2'            : (16,1),
    'int2e_ssp1sps2'            : (16,1),
    'int2e_sps1ssp2'            : (16,1),
    'int2e_sps1sps2'            : (16,1),
    'int2e_cg_ssa10ssp2'        : (48,3),
    'int2e_giao_ssa10ssp2'      : (18,3),
    'int2e_gssp1ssp2'           : (18,3),
    'int2e_gauge_r1_ssp1ssp2'   : (None, 1),
    'int2e_gauge_r1_ssp1sps2'   : (None, 1),
    'int2e_gauge_r1_sps1ssp2'   : (None, 1),
    'int2e_gauge_r1_sps1sps2'   : (None, 1),
    'int2e_gauge_r2_ssp1ssp2'   : (None, 1),
    'int2e_gauge_r2_ssp1sps2'   : (None, 1),
    'int2e_gauge_r2_sps1ssp2'   : (None, 1),
    'int2e_gauge_r2_sps1sps2'   : (None, 1),
    'int1e_ipipovlp'            : (9, 9),
    'int1e_ipovlpip'            : (9, 9),
    'int1e_ipipkin'             : (9, 9),
    'int1e_ipkinip'             : (9, 9),
    'int1e_ipipnuc'             : (9, 9),
    'int1e_ipnucip'             : (9, 9),
    'int1e_ipiprinv'            : (9, 9),
    'int1e_iprinvip'            : (9, 9),
    'int2e_ipip1'               : (9, 9),
    'int2e_ipvip1'              : (9, 9),
    'int2e_ip1ip2'              : (9, 9),
    'int1e_ipippnucp'           : (9, 9),
    'int1e_ippnucpip'           : (9, 9),
    'int1e_ipipprinvp'          : (9, 9),
    'int1e_ipprinvpip'          : (9, 9),
    'int1e_ipipspnucsp'         : (36,9),
    'int1e_ipspnucspip'         : (36,9),
    'int1e_ipipsprinvsp'        : (36,9),
    'int1e_ipsprinvspip'        : (36,9),
    'int3c2e'                   : (1, 1),
    'int3c2e_ip1'               : (3, 3),
    'int3c2e_ip2'               : (3, 3),
    'int3c2e_pvp1'              : (1, 1),
    'int3c2e_pvxp1'             : (3, 3),
    'int2c2e_ip1'               : (3, 3),
    'int2c2e_ip2'               : (3, 3),
    'int3c2e_ig1'               : (3, 3),
    'int3c2e_spsp1'             : (4, 1),
    'int3c2e_ipspsp1'           : (12,3),
    'int3c2e_spsp1ip2'          : (12,3),
    'int3c2e_ipip1'             : (9, 9),
    'int3c2e_ipip2'             : (9, 9),
    'int3c2e_ipvip1'            : (9, 9),
    'int3c2e_ip1ip2'            : (9, 9),
    'int2c2e_ip1ip2'            : (9, 9),
    'int2c2e_ipip1'             : (9, 9),
    'int3c1e'                   : (1, 1),
    'int3c1e_p2'                : (1, 1),
    'int3c1e_iprinv'            : (3, 3),
    'int2c2e'                   : (1, 1),
    'int2e_yp'                  : (1, 1),
    'int2e_stg'                 : (1, 1),
    'int2e_coulerf'             : (1, 1),
    "int1e_grids"               : (1, 1),
    "int1e_grids_ip"            : (3, 3),
    "int1e_grids_spvsp"         : (4, 1),
    'ECPscalar'                 : (1, None),
    'ECPscalar_ipnuc'           : (3, None),
    'ECPscalar_iprinv'          : (3, None),
    'ECPscalar_ignuc'           : (3, None),
    'ECPscalar_iprinvip'        : (9, None),
    'ECPso'                     : (3, 1),
}



def _stand_sym_code(sym):
    if isinstance(sym, int):
        return 's%d' % sym
    elif sym[0] in 'sS':
        return sym.lower()
    else:
        return 's' + sym.lower()

def ascint3(intor_name):
    '''convert cint2 function name to cint3 function name'''
    if intor_name.startswith('cint'):
        intor_name = intor_name[1:]
    if not intor_name.endswith(('_sph', '_cart', '_spinor', '_ssc')):
        intor_name = intor_name + '_spinor'
    return intor_name



def _get_intor_and_comp(intor_name, comp=None):
    intor_name = ascint3(intor_name)
    if comp is None:
        try:
            if '_spinor' in intor_name:
                fname = intor_name.replace('_spinor', '')
                comp = _INTOR_FUNCTIONS[fname][1]
            else:
                fname = intor_name.replace('_sph', '').replace('_cart', '')
                comp = _INTOR_FUNCTIONS[fname][0]
        except KeyError:
            warnings.warn('Function %s not found.  Set its comp to 1' % intor_name)
            comp = 1
    return intor_name, comp

truth = mol.intor("int2e_ip1")
us = getints("int2e_ip1_sph", mol._atm, mol._bas, mol._env, None, None, 0, "s1", None)


print(np.max(np.abs(truth-us)))
assert np.allclose(truth, us )
print("PASSED")

