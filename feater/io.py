import os, sys, tempfile, subprocess, shutil, hashlib

import numpy as np
import pytraj as pt
import h5py as h5

from rdkit import Chem
from rdkit.Chem import AllChem, rdMolTransforms, rdFMCS

import feater, siesta


class hdffile(h5.File):
  """
  This class is a wrapper of h5py.File which enables the "with" statement
  Wrapped up functions to read/write point data to HDF file
  """
  def __init__(self, filename, reading_flag):
    # print(f"Opening HDF5 file '{filename}' in mode '{reading_flag}'")
    super().__init__(filename, reading_flag)

  # __enter__ and __exit__ are used to enable the "with" statement
  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    self.close()

  def data(self, key):
    dset = self[key]
    return np.asarray(dset)

  def dtype(self,key):
    dset = self[key]
    return dset.dtype

  def draw_structure(self, depth=0):
    """
    Draw the structure of the HDF file. By default, only display the first level of the file (depth=0).
    Args:
      depth (int): The depth of the file structure to display.
    """
    print("############## HDF File Structure ##############")
    def print_structure(name, obj):
      current_depth = name.count('/')
      if current_depth > depth:  # If the current item's depth exceeds the limit, return early
        return
      if isinstance(obj, h5.Group):
        print(f"$ /{name:20s}/Heterogeneous Group")
      else:
        print(f"$ /{name:20s}: Shape-{obj.shape}")

    self.visititems(print_structure)
    print("############ END HDF File Structure ############")

  def append_entries(self, dataset_name:str, data:np.ndarray):
    """
    Append entries to an existing dataset.
    Args:
      dataset_name (str): The name of the dataset to append to.
      data (np.ndarray): The data to append to the dataset.
    NOTE:
      When creating the dataset, make sure to set maxshape to [None, 3]
    """
    dset = self[dataset_name]
    current_shape = dset.shape
    # Calculate the new shape after appending the new data
    new_shape = (current_shape[0] + data.shape[0], *current_shape[1:])
    # Resize the dataset to accommodate the new data
    dset.resize(new_shape)
    # Append the new data to the dataset
    dset[current_shape[0]:new_shape[0]] = data


class StructureProcessor:
  """
  This class is used to process the sub-structure of a protein
  It registers the rules on how to process the sub-structures and with do_prod() to start the processing
  """
  def __init__(self):
    self.sequence = []
    self.topology = pt.Topology()
    self.coordinates = np.array([])
    self.active_residue = 0
    self.nr_residues = 0
    self.nr_frames = 0

    # Output settings
    self.output_settings = {
      "write_pdb": False,
      "write_sdf": False,
      "write_mol2": False,
      "write_surf": False,
      "output_folder": "/tmp/"
    }
    self._output_folder = os.path.abspath(self.output_settings["output_folder"])
    self._write_pdb = self.output_settings["write_pdb"]
    self._write_sdf = self.output_settings["write_sdf"]
    self._write_mol2 = self.output_settings["write_mol2"]
    self._write_surf = self.output_settings["write_surf"]
    self._focused_residue = None

  def read_structure(self, filename):
    thetraj = pt.load(filename)
    self.topology = thetraj.top
    self.coordinates = thetraj.xyz
    self.nr_residues = self.topology.n_residues
    self.nr_frames = self.coordinates.shape[0]
    self.sequence = [i for i in self.topology.residues]
    self.atoms = [i for i in self.topology.atoms]   # TODO: remove this
    if len(self.sequence) != self.nr_residues:
      raise ValueError(f"Number of residues {self.nr_residues} does not match the sequence length {len(self.sequence)}")

  def focus_residue(self, residue_name):
    self._focused_residue = residue_name

  def next(self):
    self.active_residue += 1

  def reset(self):
    self.active_residue = 0

  def update_config(self, **kwargs):
    if "write_pdb" in kwargs:
      self.output_settings["write_pdb"] = kwargs["write_pdb"]
    if "write_sdf" in kwargs:
      self.output_settings["write_sdf"] = kwargs["write_sdf"]
    if "write_mol2" in kwargs:
      self.output_settings["write_mol2"] = kwargs["write_mol2"]
    if "write_surf" in kwargs:
      self.output_settings["write_surf"] = kwargs["write_surf"]
    if "output_folder" in kwargs:
      outfolder = kwargs["output_folder"]
      self.output_settings["output_folder"] = outfolder
      if not os.path.isdir(outfolder):
        raise ValueError(f"Output folder {outfolder} does not exist")
    self._output_folder = os.path.abspath(self.output_settings["output_folder"])
    self._write_pdb = self.output_settings["write_pdb"]
    self._write_sdf = self.output_settings["write_sdf"]
    self._write_mol2 = self.output_settings["write_mol2"]
    self._write_surf = self.output_settings["write_surf"]

  def process_residue_3letter(self, residue):
    os.makedirs(os.path.abspath(os.path.abspath(os.path.join(self._output_folder, residue))), exist_ok=True)
    for f_idx in np.arange(self.nr_frames):
      for r_idx in np.arange(self.nr_residues):
        current_residue = self.sequence[r_idx]
        if current_residue.name.upper() != residue.upper():
          continue
        # atom_indices = np.arange(current_residue.first_atom_index, current_residue.last_atom_index)
        sub_top = self.topology[current_residue.first_atom_index:current_residue.last_atom_index]
        sub_xyz = self.coordinates[f_idx, current_residue.first_atom_index:current_residue.last_atom_index, :]

        # Write out the temporary residue, and use CAMAPRI to fix the broken residues
        with tempfile.TemporaryDirectory() as tempdir:
          tempf_pdb = os.path.join(tempdir, "tmp.pdb")
          tempf_key = os.path.join(tempdir, "tmp.key")
          tempf_seq = os.path.join(tempdir, "tmp.seq")
          tmp_traj = pt.Trajectory(top=sub_top, xyz=np.array([sub_xyz]))
          tmp_traj.save(tempf_pdb, overwrite=True)

          # Use CAMAPRI to fix the broken residues
          with open(tempf_seq, "w") as tempf:
            if current_residue.name in HOMOGENEOUS_LIST:
              tempf.write(f"{current_residue.name}_N_C\nEND\n")
            elif current_residue.name == "HIS":
              tempf.write(f"HID_N_C\nEND\n")
            elif current_residue.name in HETEROGENEOUS_LIST:
              # TODO; Skip temporarily and implement this later
              # TODO: HETEROGENEOUS_LIST = ['ASP', 'CYS', 'GLU', 'HIS', 'LYS']
              tempf.write(f"{current_residue.name}_N_C\nEND\n")
          output_pdb = feater.scripts.fix_residue.fix_residue(tempdir,
                                                  seq_file = tempf_seq,
                                                  pdb_file = tempf_pdb,
                                                  key_file = tempf_key)
          if len(output_pdb) == 0:
            print(f"CAMPARI failes to fix the structure of residue {current_residue.name}")
            continue

          molhash = hashlib.md5(output_pdb["result_pdb"].encode()).hexdigest()[:8]
          try:
            themol = Chem.MolFromPDBBlock(output_pdb["result_pdb"], sanitize=True, removeHs=False)
            themol.SetProp("_Name", f"{current_residue.name}_{molhash}")
            heavy_atom_number = len([i for i in themol.GetAtoms() if i.GetAtomicNum() > 1])
          except:
            print(f"Residue {current_residue.name} cannot be converted to RDKit mol")
            continue

          if f"{current_residue.name}_{heavy_atom_number}" not in FREQ_CONFIG:
            print(f"Residue {current_residue.name} has {heavy_atom_number} heavy atoms")
            continue
          else:
            # print("-s", end="")
            pass

          tmp_prefix = os.path.join(tempdir, "tmpfile")
          output_prefix = os.path.join(self._output_folder, current_residue.name,
                                       f"{current_residue.name}_{heavy_atom_number}_{molhash}")
          if self._write_pdb:
            with open(f"{tmp_prefix}.pdb", "w") as f:
              f.write(Chem.MolToPDBBlock(themol))
          if self._write_mol2:
            write_mol2(themol, f"{tmp_prefix}.mol2", resname=current_residue.name)
          if self._write_sdf:
            Chem.MolToMolFile(themol, f"{tmp_prefix}.sdf")
          if self._write_surf:
            mol_to_surf(themol, f"{tmp_prefix}.ply")
          if os.path.exists(f"{tmp_prefix}.pdb"):
            shutil.copy2(f"{tmp_prefix}.pdb", f"{output_prefix}.pdb")
          if os.path.exists(f"{tmp_prefix}.mol2"):
            shutil.copy2(f"{tmp_prefix}.mol2", f"{output_prefix}.mol2")
          if os.path.exists(f"{tmp_prefix}.sdf"):
            shutil.copy2(f"{tmp_prefix}.sdf", f"{output_prefix}.sdf")
          if os.path.exists(f"{tmp_prefix}.ply"):
            shutil.copy2(f"{tmp_prefix}.ply", f"{output_prefix}.ply")
    print("\n_______________________")

def mol_to_surf(mol, output_file):
  xyzr_arr = np.zeros((mol.GetNumAtoms(), 4), dtype=np.float32)
  conf = mol.GetConformer()
  for i in range(mol.GetNumAtoms()):
    pos = conf.GetAtomPosition(i)
    xyzr_arr[i, 0] = pos.x
    xyzr_arr[i, 1] = pos.y
    xyzr_arr[i, 2] = pos.z
    xyzr_arr[i, 3] = 1
  siesta.xyzr_to_file(xyzr_arr, output_file, format="ply", grid_size=0.2, slice_number=300, smooth_step=0)
  return xyzr_arr

def write_mol2(mol, filename, resname='UNL'):
  AllChem.ComputeGasteigerCharges(mol)
  # Get the Gasteiger charge of each atom in the molecule
  charges = [round(float(atom.GetProp("_GasteigerCharge")),6) for atom in mol.GetAtoms()]
  with open(filename, 'w') as f:
    # Write the Mol2 header
    f.write('@<TRIPOS>MOLECULE\n')
    f.write(mol.GetProp("_Name")+'\n')
    f.write('%d %d 0 0 0\n' % (mol.GetNumAtoms(), mol.GetNumBonds()))
    f.write('SMALL\nGASTEIGER\n\n')
    # Write the atom block
    f.write('@<TRIPOS>ATOM\n')
    for i, atom in enumerate(mol.GetAtoms()):
      pos = mol.GetConformer().GetAtomPosition(i)
      f.write(f'{i+1} {atom.GetSymbol()}{i+1} {pos.x:.4f} {pos.y:.4f} {pos.z:.4f} {_sybyl_atom_type(atom)} 1 {resname} {charges[i]}\n')
    # Write the bond block
    f.write('@<TRIPOS>BOND\n')
    for bond in mol.GetBonds():
      f.write('%d %d %d %s\n' % (bond.GetIdx()+1, bond.GetBeginAtomIdx()+1, bond.GetEndAtomIdx()+1, 'ar' if bond.GetIsAromatic() else int(bond.GetBondTypeAsDouble())))

def _sybyl_atom_type(atom):
  """
  Code piece from Open Drug Discovery Toolkit (URL https://oddt.readthedocs.io/en/latest/_modules/oddt/toolkits/extras/rdkit.html)
  Asign sybyl atom types for mol2 writer from rdkit molecule
  Reference #1: http://www.tripos.com/mol2/atom_types.html
  Reference #2: http://chemyang.ccnu.edu.cn/ccb/server/AIMMS/mol2.pdf
  """
  sybyl = None
  atom_symbol = atom.GetSymbol()
  atomic_num = atom.GetAtomicNum()
  hyb = atom.GetHybridization()-1  # -1 since 1 = sp, 2 = sp1 etc
  hyb = min(hyb, 3)
  degree = atom.GetDegree()
  aromtic = atom.GetIsAromatic()
  # define groups for atom types
  guanidine = '[NX3,NX2]([!O,!S])!@C(!@[NX3,NX2]([!O,!S]))!@[NX3,NX2]([!O,!S])'  # strict
  if atomic_num == 6:
    if aromtic:
      sybyl = 'C.ar'
    elif degree == 3 and _atom_matches_smarts(atom, guanidine):
      sybyl = 'C.cat'
    else:
      sybyl = '%s.%i' % (atom_symbol, hyb)
  elif atomic_num == 7:
    if aromtic:
      sybyl = 'N.ar'
    elif _atom_matches_smarts(atom, 'C(=[O,S])-N'):
      sybyl = 'N.am'
    elif degree == 3 and _atom_matches_smarts(atom, '[$(N!-*),$([NX3H1]-*!-*)]'):
      sybyl = 'N.pl3'
    elif _atom_matches_smarts(atom, guanidine):  # guanidine has N.pl3
      sybyl = 'N.pl3'
    elif degree == 4 or hyb == 3 and atom.GetFormalCharge():
      sybyl = 'N.4'
    else:
      sybyl = '%s.%i' % (atom_symbol, hyb)
  elif atomic_num == 8:
    # http://www.daylight.com/dayhtml_tutorials/languages/smarts/smarts_examples.html
    if degree == 1 and _atom_matches_smarts(atom, '[CX3](=O)[OX1H0-]'):
      sybyl = 'O.co2'
    elif degree == 2 and not aromtic:  # Aromatic Os are sp2
      sybyl = 'O.3'
    else:
      sybyl = 'O.2'
  elif atomic_num == 16:
    # http://www.daylight.com/dayhtml_tutorials/languages/smarts/smarts_examples.html
    if degree == 3 and _atom_matches_smarts(atom, '[$([#16X3]=[OX1]),$([#16X3+][OX1-])]'):
      sybyl = 'S.O'
    # https://github.com/rdkit/rdkit/blob/master/Data/FragmentDescriptors.csv
    elif _atom_matches_smarts(atom, 'S(=,-[OX1;+0,-1])(=,-[OX1;+0,-1])(-[#6])-[#6]'):
      sybyl = 'S.o2'
    else:
      sybyl = '%s.%i' % (atom_symbol, hyb)
  elif atomic_num == 15 and hyb == 3:
    sybyl = '%s.%i' % (atom_symbol, hyb)
  if not sybyl:
    sybyl = atom_symbol
  return sybyl

def _atom_matches_smarts(atom, smarts):
  """
  Match substructures for atom type assignment
  """
  idx = atom.GetIdx()
  patt = Chem.MolFromSmarts(smarts)
  for m in atom.GetOwningMol().GetSubstructMatches(patt):
    if idx in m:
      return True
  return False



FREQ_CONFIG = [
  "GLY_5",
  "ALA_6",
  "CYS_7",  # 10, 11 and 13
  "SER_7",  # 11 (most) and 13, 15, 9
  "THR_8",  # 14
  "PRO_8",  # 14 (most) and 15
  "VAL_8",  # 16

  "MET_9",  # 17 (most) and 19
  "ASN_9",  # 14 (most) and 9
  "ASP_9",  # 12
  "LEU_9",  # 19
  "ILE_9",  # 19 (most) and 21
  "LYS_10",  # 22
  "GLN_10",  # 17 (most) and 18, 19, 9
  "GLU_10",  # 15 (most) and 9, 17

  "HIS_11",  # 16 (most) and 9
  "ARG_12",  # 24
  "PHE_12",  # 20 (most) and 21, 19
  "TYR_13",  # 21
  "TRP_15",  # 24
]

# HOMOGENEOUS = {
#   "ALA": 1,
#   "ARG": 1,
#   "ASN": 1,
#   "GLN": 1,
#   "GLY": 1,
#   "ILE": 1,
#   "LEU": 1,
#   "MET": 1,
#   "PHE": 1,
#   "PRO": 1,
#   "SER": 1,
#   "THR": 1,
#   "TRP": 1,
#   "TYR": 1,
#   "VAL": 1
# }

HOMOGENEOUS_LIST = ['ALA', 'ARG', 'ASN', 'GLN', 'GLY', 'ILE', 'LEU', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']
HETEROGENEOUS_LIST = ['ASP', 'CYS', 'GLU', 'HIS', 'LYS']

# HETEROGENEOUS = {
#   "ASP": 1, # ASH, ASP
#   "CYS": 2, # CYS, CYX
#   "GLU": 1, # GLH, GLU
#   "HIS": 3, # HID, HIE, HIP ? Very special case
#   "LYS": 1, # LYN, LYS
# }

