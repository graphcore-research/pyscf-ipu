# produce a sorted version of GDB9, sorted by the number of hydrogens when using rdkit. 

smiles = [a.split("\t")[0] for a in open("gdb11_size09.smi", "r").read().split("\n")]


num_hs = []
for smile in smiles:
  atoms = [a for a in list(smile) if a == "C" or a == "N" or a == "O" or a == "F"]
  from rdkit import Chem  
  from rdkit.Chem import AllChem

  from rdkit import RDLogger
  lg = RDLogger.logger()
  lg.setLevel(RDLogger.CRITICAL)

  b = Chem.MolFromSmiles(smile)
  # this add ~ 10 hydrogens or so ==;; 55 => 71
  b = Chem.AddHs(b)  # perhaps more numerically stable wiht the H's? e.g. energy may be lower?

  AllChem.EmbedMolecule(b)
  # just define a function for each numbre of hydrogens and then index into that list whenever we run! 
  # compiled_funcs[num_hydrogens] = ...

  atoms = [atom.GetSymbol() for atom in b.GetAtoms()]
  num_hs.append( len([a for a in atoms if a == "H"]))

smiles = smiles()