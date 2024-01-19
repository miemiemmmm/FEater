import os, sys, time, argparse, json, tempfile, hashlib, shutil

import multiprocessing as mp
import pytraj as pt
import numpy as np

# import feater
from feater.scripts import fix_residue
from feater import utils, constants


def parser_run_pdbprocess():
  parser = argparse.ArgumentParser(description="Run the automatic pdb processing to establish the FEater database")
  parser.add_argument("-i", "--input", default="", type=str, help="Input protein file list. ")
  parser.add_argument("-d", "--directory", default="", type=str, help="Output directory; New folders named by the residues will be built in this folder. ")
  parser.add_argument("-prod", "--production_mode", default=0, type=int, help="Run the production mode; Only explicitly set to 1 to run the production mode, otherwise run the test mode; Default: 0.")
  parser.add_argument("-pn", "--thread_number", default=4, type=int, help="Number of threads; Default: 4. ")

  parser.add_argument("--node_number", default=1, type=int, help="Only used in the slurm mode.")
  parser.add_argument("--node_index", default=0, type=int, help="Only used in the slurm mode.")

  args = parser.parse_args()

  camp_dir = os.environ.get("CAMPARI_DIR", "")
  camp_exe = os.environ.get("CAMPARI_EXE", "")

  if camp_dir == "" or camp_exe == "":
    raise ValueError("Please define the environment variables: CAMPARI_DIR, CAMPARI_EXE")
  if not os.path.exists(camp_dir):
    raise FileNotFoundError(f"Cannot find the campari directory: {camp_dir}")
  if not os.path.exists(camp_exe):
    raise FileNotFoundError(f"Cannot find the campari executable: {camp_exe}")
  args.campari_dir = camp_dir
  args.campari_exe = camp_exe

  # Check the existence of input file list and output directory
  if not os.path.exists(args.input):
    print(f"Cannot find the input file list: {args.input}", file=sys.stderr)
    parser.print_help()
    exit(1)
  if not os.path.exists(args.directory):
    print(f"Cannot find the output directory: {args.directory}", file=sys.stderr)
    parser.print_help()
    exit(1)

  return args


def process_file_dual(complex_file: str, settings: dict):
  print(f"Processing the PDB file: {complex_file}")
  st = time.perf_counter()
  peptide_bond_threshold = 1.32*1.25
  traj = pt.load(complex_file)
  coord = traj.xyz[0]
  residues = [res for res in traj.top.residues]
  for i, res1 in enumerate(residues[:-1]):
    res2 = residues[i+1]
    top_res1 = traj.top[res1.first_atom_index:res1.last_atom_index]
    crd_res1 = coord[res1.first_atom_index:res1.last_atom_index, :]
    top_res2 = traj.top[res2.first_atom_index:res2.last_atom_index]
    crd_res2 = coord[res2.first_atom_index:res2.last_atom_index, :]
    atomc = top_res1.select("@C")
    atomn = top_res2.select("@N")
    if (res1.name not in constants.RES) or (res2.name not in constants.RES):
      print(f"Residue pair {res1.name}-{res2.name} is not supported yet", file=sys.stderr)
      continue

    if len(atomc) == 0 or len(atomn) == 0:
      continue
    elif len(atomc) > 1 or len(atomn) > 1:
      print("Warning: More than one C or N atom found in the residue, meaning there are some problem with the input structure.")
      continue
    else:
      peptide_bond_len = np.linalg.norm(crd_res1[atomc[0]] - crd_res2[atomn[0]])
      if peptide_bond_len > peptide_bond_threshold:
        continue

      sub_top = traj.top[res1.first_atom_index:res2.last_atom_index]
      sub_xyz = traj.xyz[0, res1.first_atom_index:res2.last_atom_index, :]
      with tempfile.TemporaryDirectory() as tempdir:
        tempf_pdb = os.path.join(tempdir, "tmp.pdb")
        tempf_key = os.path.join(tempdir, "tmp.key")
        tempf_seq = os.path.join(tempdir, "tmp.seq")
        tmp_traj = pt.Trajectory(top=sub_top, xyz=np.array([sub_xyz]))
        tmp_traj.save(tempf_pdb, overwrite=True)

        with open(tempf_seq, "w") as tempf:
          if res1.name == "HIS":
            r1name = "HID"
          else:
            r1name = res1.name
          if res2.name == "HIS":
            r2name = "HID"
          else:
            r2name = res2.name
          tempf.write(f"{r1name}_N\n")
          tempf.write(f"{r2name}_C\n")
          tempf.write("END\n")
        output_pdb = fix_residue.fix_residue(tempdir,
                                             seq_file=tempf_seq,
                                             pdb_file=tempf_pdb,
                                             key_file=tempf_key)
        if len(output_pdb) == 0:
          print("Residue fixer failed", file=sys.stderr)
          continue
        molhash = hashlib.md5(output_pdb["result_pdb"].encode()).hexdigest()[:8]
        tmp_prefix = os.path.join(tempdir, "tmpfile")
        type_name = f"{res1.name}{res2.name}"
        with open(f"{tmp_prefix}.pdb", "w") as tmp_pdb:
          tmp_pdb.write(output_pdb["result_pdb"])
        output_prefix = os.path.join(os.path.abspath(settings["directory"]), type_name, f"{type_name}_{molhash}")
        if not os.path.exists(os.path.join(os.path.abspath(settings["directory"]), type_name)):
          os.mkdir(os.path.join(os.path.abspath(settings["directory"]), type_name))
        if os.path.exists(f"{tmp_prefix}.pdb"):
          shutil.copy2(f"{tmp_prefix}.pdb", f"{output_prefix}.pdb")
  print(f"Finished the processing of the PDB file: {complex_file}; Time elapsed: {(time.perf_counter() - st):.2f} seconds")


def do_prod_slurm(complex_files: list, args: dict):
  # Retrieve the list of complexes to process based on the array task index
  files_todo = np.array_split(complex_files, args["node_number"])[args["node_index"]]
  print(complex_files)
  print(files_todo)
  print("#"*80)
  print(f"Node index: {args['node_index']}/{args['node_number']}")
  print(f"This task is process: {len(files_todo)}/{len(complex_files)} complexes")

  
  # Initialize the thread pool and prepare the tasks for the thread pool
  thread_pool = mp.Pool(args["thread_number"])
  tasks = [(pdb, args) for pdb in files_todo]
  thread_pool.starmap(process_file_dual, tasks)

  print(f"Finished the dual residue processing of {len(files_todo)} complexes")
  print("^"*80)
  thread_pool.close()
  thread_pool.join()


def run_pdbprocess():
  args = parser_run_pdbprocess()
  settings = json.dumps(vars(args), indent=2)
  print("Settings: ", settings)
  complex_list = utils.checkfiles(args.input)  #[:50]    # TODO: Remove the slice for production
  print("Total number of complexes: ", len(complex_list))
  do_prod_slurm(complex_list, vars(args))



if __name__ == "__main__":
  run_pdbprocess()
