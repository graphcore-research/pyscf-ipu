# reproduce qm9 labels; run download.sh to download qm9 first. 
from pyscf import gto, scf, dft, __config__
import pyscf
import pandas as pd 
print(pyscf.__version__)
df = pd.read_csv('qm9.csv')
qm9_energy = df['u0'][0] - df['zpve'][0]
qm9_hlgap = df['gap'][0]

mol = gto.Mole()
mol.atom = '''
    C   -0.0127    1.0858    0.0080
    H    0.0022   -0.0060    0.0020
    H    1.0117    1.4638    0.0003
    H   -0.5408    1.4475   -0.8766
    H   -0.5238    1.4379    0.9064
'''
mol.basis = '6-31G(2df,p)'
mol.build()

# Run B3LYP calculation
method = dft.RKS(mol)
method.verbose = 4
method.xc = 'B3LYPG' # b3lypG (G as in gaussain)
method.max_cycle = 50 
method.DIIS = pyscf.scf.diis.CDIIS
method.small_rho_cutoff = 1e-10
method.diis_space = 8
method.diis_start_cycle = 1 
method.damp = 5e-1 # damping factor
method.conv_tol = 1e-9
method.conv_tol_grad = None # 1e-9
method.grids.level = 3
method.kernel()

# Get total energy and HOMO-LUMO gap
energy = method.e_tot
homo, lumo = method.mo_energy[method.mo_occ>0].max(), method.mo_energy[method.mo_occ==0].min()
hlgap = lumo - homo

print('qm9\t %10f %10f'%(qm9_energy, qm9_hlgap))
print('pyscf\t %10f %10f'%( energy, hlgap))
