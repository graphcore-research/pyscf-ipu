import rdkit
from rdkit.Chem import AllChem

from rdkit import Chem
from rdkit.Chem import Descriptors
from multiprocessing import Pool
from rdkit.Chem import EnumerateStereoisomers
import pandas as pd 
from tqdm import tqdm 


import sys 
number = int(sys.argv[1])

smiles = pd.read_csv("/a/scratch/alexm/research/gdb/gdb13/gdb13.cno._%02i.csv"%number, header=None) # this took 5s to load? 
smiles = smiles[0].values

# count hydrogens and flippers 
# make this into a script and just run it a shit ton of times. 
# flippers are the ones we use to sort before doing somers to have load balancing for parallel processing. 

# all of this string stuff; if we tilejax this basic cheminformatics one IPU could do this 
from rdkit.Chem.EnumerateStereoisomers import * 
from rdkit.Chem.EnumerateStereoisomers import * 



from rdkit.Chem.EnumerateStereoisomers import * 

class _BondFlipper(object):

  def __init__(self, bond):
    self.bond = bond

  def flip(self, flag):
    if flag:
      self.bond.SetStereo(Chem.BondStereo.STEREOCIS)
    else:
      self.bond.SetStereo(Chem.BondStereo.STEREOTRANS)


class _AtomFlipper(object):

  def __init__(self, atom):
    self.atom = atom

  def flip(self, flag):
    if flag:
      self.atom.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW)
    else:
      self.atom.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CCW)


class _StereoGroupFlipper(object):

  def __init__(self, group):
    self._original_parities = [(a, a.GetChiralTag()) for a in group.GetAtoms()]

  def flip(self, flag):
    if flag:
      for a, original_parity in self._original_parities:
        a.SetChiralTag(original_parity)
    else:
      for a, original_parity in self._original_parities:
        if original_parity == Chem.ChiralType.CHI_TETRAHEDRAL_CW:
          a.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
        elif original_parity == Chem.ChiralType.CHI_TETRAHEDRAL_CCW:
          a.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW)


def _getFlippers(mol, options):
  Chem.FindPotentialStereoBonds(mol)
  flippers = []
  if not options.onlyStereoGroups:
    for atom in mol.GetAtoms():
      if atom.HasProp("_ChiralityPossible"):
        if (not options.onlyUnassigned or atom.GetChiralTag() == Chem.ChiralType.CHI_UNSPECIFIED):
          flippers.append(_AtomFlipper(atom))

    for bond in mol.GetBonds():
      bstereo = bond.GetStereo()
      if bstereo != Chem.BondStereo.STEREONONE:
        if (not options.onlyUnassigned or bstereo == Chem.BondStereo.STEREOANY):
          flippers.append(_BondFlipper(bond))

  if options.onlyUnassigned:
    # otherwise these will be counted twice
    for group in mol.GetStereoGroups():
      if group.GetGroupType() != Chem.StereoGroupType.STEREO_ABSOLUTE:
        flippers.append(_StereoGroupFlipper(group))

  return flippers


class _RangeBitsGenerator(object):

  def __init__(self, nCenters):
    self.nCenters = nCenters

  def __iter__(self):
    for val in range(2**self.nCenters):
      yield val


class _UniqueRandomBitsGenerator(object):

  def __init__(self, nCenters, maxIsomers, rand):
    self.nCenters = nCenters
    self.maxIsomers = maxIsomers
    self.rand = rand
    self.already_seen = set()

  def __iter__(self):
    # note: important that this is not 'while True' otherwise it
    # would be possible to have an infinite loop caused by all
    # isomers failing the embedding process
    while len(self.already_seen) < 2**self.nCenters:
      bits = self.rand.getrandbits(self.nCenters)
      if bits in self.already_seen:
        continue

      self.already_seen.add(bits)
      yield bits


def GetStereoisomerCount(m, options=StereoEnumerationOptions()):
  """ returns an estimate (upper bound) of the number of possible stereoisomers for a molecule
   Arguments:
      - m: the molecule to work with
      - options: parameters controlling the enumeration
    >>> from rdkit import Chem
    >>> from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions
    >>> m = Chem.MolFromSmiles('BrC(Cl)(F)CCC(O)C')
    >>> GetStereoisomerCount(m)
    4
    >>> m = Chem.MolFromSmiles('CC(Cl)(O)C')
    >>> GetStereoisomerCount(m)
    1
    double bond stereochemistry is also included:
    >>> m = Chem.MolFromSmiles('BrC(Cl)(F)C=CC(O)C')
    >>> GetStereoisomerCount(m)
    8
    """
  tm = Chem.Mol(m)
  flippers = _getFlippers(tm, options)
  return 2**len(flippers)


# benchmark this script, which parts of it take time? 
def CustomEnumerateStereoisomers(m, max=9, options=StereoEnumerationOptions(), verbose=False):
  # it takes linearly longer in the number of isomers; so the stuff that takes time in num_atoms/num_bonds 
  # is likely not dominating! 
  #tm = Chem.MolFromSmiles(m)
  tm = m 

  for atom in tm.GetAtoms(): # loop through atoms 
    atom.ClearProp("_CIPCode")
  for bond in tm.GetBonds(): # loop through bonds 
    if bond.GetBondDir() == Chem.BondDir.EITHERDOUBLE:
      bond.SetBondDir(Chem.BondDir.NONE)
  flippers = _getFlippers(tm, options)
  nCenters = len(flippers)
  if not nCenters:
    yield tm
    return

  #q: what does it use the random bits for? can we generate these with numpy and pass as input? perhaps that's the bottleneck? 

  if (options.maxIsomers == 0 or 2**nCenters <= options.maxIsomers):
    bitsource = _RangeBitsGenerator(nCenters)
  else:
    if options.rand is None:
      # deterministic random seed invariant to input atom order
      seed = hash(tuple(sorted([(a.GetDegree(), a.GetAtomicNum()) for a in tm.GetAtoms()])))
      rand = random.Random(seed)
    elif isinstance(options.rand, random.Random):
      # other implementations of Python random number generators
      # can inherit from this class to pick up utility methods
      rand = options.rand
    else:
      rand = random.Random(options.rand)
    bitsource = _UniqueRandomBitsGenerator(nCenters, options.maxIsomers, rand)

  isomersSeen = set()
  numIsomers = 0

  for bitflag in bitsource:
    for i in range(nCenters):
      flag = bool(bitflag & (1 << i))
      flippers[i].flip(flag)

    # from this point on we no longer need the stereogroups (if any are there), so
    # remove them:
    if tm.GetStereoGroups():
      isomer = Chem.RWMol(tm)
      isomer.SetStereoGroups([])
    else:
      isomer = Chem.Mol(tm)
    Chem.SetDoubleBondNeighborDirections(isomer)
    isomer.ClearComputedProps(includeRings=False)

    Chem.AssignStereochemistry(isomer, cleanIt=True, force=True, flagPossibleStereoCenters=True)
    if options.unique:
      cansmi = Chem.MolToSmiles(isomer, isomericSmiles=True)
      if cansmi in isomersSeen:
        continue

      isomersSeen.add(cansmi)

    if options.tryEmbedding:
      ntm = Chem.AddHs(isomer)
      # mask bitflag to fit within C++ int.
      cid = EmbedMolecule(ntm, randomSeed=(bitflag & 0x7fffffff))
      if cid >= 0:
        conf = Chem.Conformer(isomer.GetNumAtoms())
        for aid in range(isomer.GetNumAtoms()):
          conf.SetAtomPosition(aid, ntm.GetConformer().GetAtomPosition(aid))
        isomer.AddConformer(conf)
    else:
      cid = 1
    if cid >= 0:
      yield isomer
      numIsomers += 1
      if options.maxIsomers != 0 and numIsomers >= options.maxIsomers or numIsomers > max:
        break
    elif verbose:
      print("%s    failed to embed" % (Chem.MolToSmiles(isomer, isomericSmiles=True)))


import numpy as np 
options = StereoEnumerationOptions()
def calculate_hs(smile):
    b = Chem.MolFromSmiles(smile)
    isomers= CustomEnumerateStereoisomers(b)
    isomers = [Chem.MolToSmiles(isomer, isomericSmiles=True) for isomer in isomers] #isomericSmiles adds [ ] brackets which contain information on stereochemistry! 
    b = Chem.AddHs(b)  # this could perhaps directly return hs and be faster because of that? 
    atoms = [atom.GetSymbol() for atom in b.GetAtoms()] 
    hs = len([a for a in atoms if a.upper() == "H"])
    #flippers = _getFlippers(b)
    #return hs, len(flippers)
    return isomers, hs

def calculate_hs2(smile):
    b = Chem.MolFromSmiles(smile)
    b = Chem.AddHs(b)  # this could perhaps directly return hs and be faster because of that?  ;; perhaps we just have to move this line above? 

    atoms = [atom.GetSymbol() for atom in b.GetAtoms()] 
    hs = len([a for a in atoms if a.upper() == "H"])
    return hs




# ~ 5 min to compute the hydrogen and flippers, not too bad! 
# can we just remove the ones with too many flippers aswell? 
# I guess we only need ~3x here, so that'd be fine? 

# more more work inside each thread so they don't have to wait on each other? 
# perhaps use pytorch dataloader for this? 
# can we get this to run efficiently within a numa node from python? 

# this is harder than anticipated, super fucking annoying... 


# create program that dumps data into file, monitors number of points in wandb, and automatically runs pyscf on random subset. 

num_hs   = []
num_flips = []
chunk = 1
isomers = []
isocount = []
for i in tqdm(range(0, len(smiles), chunk)): # 100*250 = 25000k/s ; i think this is similar to not using the threads? 
  current   = smiles[i:i+chunk]
  _hs = calculate_hs2( current[0]) 
  num_hs.append(_hs) 

  if i % 10000 == 0: 
      df = pd.DataFrame({"smiles": isomers, "hs": num_hs})
      df.to_parquet("%02i_v3.parquet"%number, compression="snappy")


df = pd.DataFrame({"smiles": isomers, "hs": num_hs})
df.to_parquet("%02i_v3.parquet"%number, compression="snappy")
