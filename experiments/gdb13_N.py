import pandas as pd 
import re 
import numpy as np 
import pyscf 
from tqdm import tqdm 
#molecules = pd.read_pickle("../data/unique.pkl")
    
def get_atom_string(atoms, locs):
    atom_string = atoms 
    atoms = re.findall('[a-zA-Z][^A-Z]*', atoms)
    str = ""
    for atom, loc in zip(atoms, locs): 
      str += "%s %4f %4f %4f; "%((atom,) + tuple(loc) )
    return atom_string, str 

def build(self, dump_input=True, parse_arg=True,
              verbose=None, output=None, max_memory=None,
              atom=None, basis=None, unit=None, nucmod=None, ecp=None,
              charge=None, spin=0, symmetry=None, symmetry_subgroup=None,
              cart=None, magmom=None): # reduced time to generate from 150 ms to 1ms by removing gc.collect. could call it each 1000 molecules or so. 

        #if sys.version_info >= (3,):
        unicode = str
        #print("ASD")
        #exit()
        from pyscf import __config__ 
        DISABLE_GC = getattr(__config__, 'DISABLE_GC', False)

        #if not DISABLE_GC and False:
        #    gc.collect()  # To release circular referred objects
        #    pass 

        if isinstance(dump_input, (str, unicode)):
            sys.stderr.write('Assigning the first argument %s to mol.atom\n' %
                             dump_input)
            dump_input, atom = True, dump_input

        if verbose is not None: self.verbose = verbose
        if output is not None: self.output = output
        if max_memory is not None: self.max_memory = max_memory
        if atom is not None: self.atom = atom
        if basis is not None: self.basis = basis
        if unit is not None: self.unit = unit
        if nucmod is not None: self.nucmod = nucmod
        if ecp is not None: self.ecp = ecp
        if charge is not None: self.charge = charge
        if spin != 0: self.spin = spin
        if symmetry is not None: self.symmetry = symmetry
        if symmetry_subgroup is not None: self.symmetry_subgroup = symmetry_subgroup
        if cart is not None: self.cart = cart
        if magmom is not None: self.magmom = magmom

        def _update_from_cmdargs_(mol):
            try:
                # Detect whether in Ipython shell
                __IPYTHON__  # noqa:
                return
            except Exception:
                pass

            if not mol._built: # parse cmdline args only once
                opts = cmd_args.cmd_args()

                if opts.verbose:
                    mol.verbose = opts.verbose
                if opts.max_memory:
                    mol.max_memory = opts.max_memory

                if opts.output:
                    mol.output = opts.output


        #if self.verbose >= logger.WARN:
        #    self.check_sanity()

        self._atom = self.format_atom(self.atom, unit=self.unit)
        uniq_atoms = set([a[0] for a in self._atom])

        if isinstance(self.basis, (str, unicode, tuple, list)):
            # specify global basis for whole molecule
            _basis = dict(((a, self.basis) for a in uniq_atoms))
        elif 'default' in self.basis:
            default_basis = self.basis['default']
            _basis = dict(((a, default_basis) for a in uniq_atoms))
            _basis.update(self.basis)
            del (_basis['default'])
        else:
            _basis = self.basis
        self._basis = self.format_basis(_basis)

        # TODO: Consider ECP info in point group symmetry initialization
        if self.ecp:
            # Unless explicitly input, ECP should not be assigned to ghost atoms
            if isinstance(self.ecp, (str, unicode)):
                _ecp = dict([(a, str(self.ecp))
                             for a in uniq_atoms if not is_ghost_atom(a)])
            elif 'default' in self.ecp:
                default_ecp = self.ecp['default']
                _ecp = dict(((a, default_ecp)
                             for a in uniq_atoms if not is_ghost_atom(a)))
                _ecp.update(self.ecp)
                del (_ecp['default'])
            else:
                _ecp = self.ecp
            self._ecp = self.format_ecp(_ecp)

        PTR_ENV_START   = 20
        env = self._env[:PTR_ENV_START]
        self._atm, self._bas, self._env = \
                self.make_env(self._atom, self._basis, env, self.nucmod,
                              self.nucprop)
        self._atm, self._ecpbas, self._env = \
                self.make_ecp_env(self._atm, self._ecp, self._env)

        if self.spin is None:
            self.spin = self.nelectron % 2
        else:
            # Access self.nelec in which the code checks whether the spin and
            # number of electrons are consistent.
            self.nelec

        if not self.magmom:
            self.magmom = [0.,]*self.natm
        import numpy 
        if self.spin == 0 and abs(numpy.sum(numpy.asarray(self.magmom)) - self.spin) > 1e-6:
            #don't check for unrestricted calcs.
            raise ValueError("mol.magmom is set incorrectly.")

        if self.symmetry:
            self._build_symmetry()

        #if dump_input and not self._built and self.verbose > logger.NOTE:
        #    self.dump_input()

        '''if self.verbose >= logger.DEBUG3:
            logger.debug3(self, 'arg.atm = %s', self._atm)
            logger.debug3(self, 'arg.bas = %s', self._bas)
            logger.debug3(self, 'arg.env = %s', self._env)
            logger.debug3(self, 'ecpbas  = %s', self._ecpbas)'''

        self._built = True
        return self


# So PCQ uses eV and PySCF uses Hartree 
hartree_to_eV    = 27.2114
angstrom_to_bohr = 1.88973

smiles = []
with open("../gdb/13.smi", "r") as file: 
    for i in range(1000000):
        line = next(file).strip()
        smiles.append(line)
#smiles = np.loadtxt("../13.smi", dtype="str")
#smiles = open("../gdb/gdb11_size11.smi", "r").read().split("\n")


lst_sto3g = []
lst_631g  = []
lst_631gs = []
lst_def2  = []
xs_sto3g  = []
xs_631g   = []
xs_631gs  = []
xs_def2   = []
import time 

# do for ANI 
# do for MD17
# do for 

mol = pyscf.gto.mole.Mole()
mol.build(atom="C 0 0 0; C 0 0 1;", unit="Bohr", basis="sto3g", spin=0, verbose=0) # all time goes into gc.collect! 
#for i, smile in enumerate(tqdm(a)):
#from rdkit import Chem  # To extract information of the molecules
#from rdkit.Chem import AllChem
#for i in tqdm(range(0, len(smiles), 10000)):
n = len(smiles)
for i in tqdm(range(0, n, 1000)):
  times = []
  #atoms = molecules["atom_string"][ids[i]]
  #locs  = molecules["atom_locations"][ids[i]]*angstrom_to_bohr
  #atom_string, _str = get_atom_string(atoms, locs)
  times.append(time.time())
  smile = smiles[i]

  atoms = [a for a in list(smile.upper()) if a == "C" or a == "N" or a == "O" or a == "Cl" or a == "S"] #+ ["H"]
  times.append(time.time())
  #b = Chem.MolFromSmiles(smile)
  #AllChem.EmbedMolecule(b)
  #locs = np.concatenate( (b.GetConformer().GetPositions(), np.ones((1,3))*4), axis=0)
  
  locs = np.random.uniform(0,1, (13, 3)) .tolist()
  times.append(time.time())

  atom_string, _str = get_atom_string(" ".join(atoms), locs)
  times.append(time.time())

  try: 
    #mol.build(atom=_str, unit="Bohr", basis="sto3g", spin=0, verbose=0) # all time goes into gc.collect! 
    build(mol,atom=_str, unit="Bohr", basis="sto3g", spin=0, verbose=0) # all time goes into gc.collect! 
    N = mol.nao_nr()
    lst_sto3g.append(N)
    xs_sto3g.append(i)
  except: 
    pass 

  times.append(time.time())
  try: 
    build(mol,atom=_str, unit="Bohr", basis="6-31G", spin=0, verbose=0) 
    lst_631g.append(mol.nao_nr())
    xs_631g.append(i)
  except: 
    pass 

  times.append(time.time())
  try: 
    build(mol,atom=_str, unit="Bohr", basis="6-31G*", spin=0, verbose=0) 
    lst_631gs.append(mol.nao_nr())
    xs_631gs.append(i)
  except: 
    pass 

  times.append(time.time())

  times = np.array(times)
  #print(np.around(times[1:]-times[:-1], 2))


import matplotlib.pyplot as plt 
fig, ax = plt.subplots(1,1, figsize=(4, 4))
#indxs = np.argsort(lst)
plt.plot(xs_sto3g, lst_sto3g, 'x', label="sto3g")
plt.plot(xs_631g,  lst_631g,  'x', label="6-31g")
plt.plot(xs_631gs, lst_631gs, 'x', label="6-31g(d)")
#plt.plot(xs_def2, lst_def2, 'x', label="def2-TZVPPD")

#plt.plot([0, 20000000], [140, 140], '-', label="Sparse=195 MB (Dense 1536 MB)")
#plt.plot([0, 20000000], [280, 280], '-', label="Sparse=3095 MB (Dense 24586 MB)")
#plt.set_yticklabels([0, 22000000], [0, 22000000])
plt.ylim([0, 300])

plt.legend()
plt.ylabel("GDB-13 (1 B)")
#ax.set_title("GDB-11 (22 Million)")
plt.xlabel("N: input size")
plt.tight_layout()
plt.savefig("gdb13_nao_nrs.jpg")
