import os, sys, time, argparse
from contextlib import contextmanager
import dask
from dask.distributed import Client, performance_report, LocalCluster
import numpy as np
import feater


SETTINGS = {
  "write_pdb": False,
  "write_mol2": False,
  "write_sdf": False,
  "write_surf": False,
  "output_folder": "/tmp/feater_test"
}
def update_config(**kwargs):
  global SETTINGS
  for key, value in kwargs.items():
    if key in SETTINGS:
      SETTINGS[key] = value
  print("The updated settings are:", SETTINGS)


def parallelize_traj(traj_list, focused_res, output_settings):
  print(f"Processing the residue: {focused_res}")
  traj_list = [i for i in traj_list]
  st_this = time.perf_counter()

  for complex_idx, complex_file in enumerate(traj_list):
    print(f"Processing {complex_file} ({complex_idx + 1}/{len(traj_list)}, last complex took {(time.perf_counter() - st_this):.2f} seconds)")
    st_this = time.perf_counter()
    complex_i = feater.io.StructureProcessor()
    complex_i.focus_residue(focused_res)
    complex_i.read_structure(complex_file)
    complex_i.update_config(**output_settings)
    complex_i.reset()
    complex_i.process_residue_3letter(focused_res)


def read_protein_list(protein_list_file):
  with open(protein_list_file, 'r') as f:
    protein_list = [i for i in f.read().strip("\n").split("\n") if i != ""]
  for p_file in protein_list:
    if not os.path.exists(p_file):
      raise FileNotFoundError(f"Cannot find the protein file: {p_file}")
  print("All protein files are found")
  return protein_list

@contextmanager
def write_performance_report(filename, enable=True):
  if enable:
    print(f"Writing the performance report to {filename}")
    with performance_report(filename=filename):
      yield
  else:
    print(f"Disabled the performance report")
    yield


def do_prod(complex_files: list, focused_res:list, output_settings: dict, worker_num = 32, thread_per_worker = 1, write_report = False):
  # Initialize the dask cluster
  split_groups = np.array_split(complex_files, worker_num)
  cluster = LocalCluster(n_workers=worker_num, threads_per_worker=thread_per_worker,
                         processes=True, memory_limit='2GB')
  with Client(cluster) as client:
    with write_performance_report(filename="FEater_perf_report.html", enable=write_report):
      tasks = []
      c = 0
      for trajlist in split_groups:
        for residue in focused_res:
          c += 1
          print(f"Task {c} | focused residue: {residue} | trajectories: {trajlist}")
          tasks.append(dask.delayed(parallelize_traj)(trajlist, residue, output_settings))

      print(f"# Task set contains {len(tasks)} jobs; Jobs are generated and ready to run")
      st = time.perf_counter()
      futures = client.compute(tasks)
      results = client.gather(futures)
      [i.release() for i in futures]
      print(f"# Task finished, time elapsed: {(time.perf_counter() - st) / 60:.2f} min")

def do_debug(complex_files: list, focused_res: str, output_settings: dict):
  start_time = time.perf_counter()
  parallelize_traj(complex_files, focused_res, output_settings)
  print(f"##### Total time elapsed: {(time.perf_counter() - start_time) / 60:.2f} min")


# Needed environment variables for feater
# During fixing the residue:
# CAMPARIDIR: The directory of the campari installation
# CAMPARI_EXE: The path to the campari executable


# Needed to defined with running the script:
# output_folder: The folder to store the output files  -d in parser
# FEATER_TEST: If set to 1, only process 30 complexes  if --prod is not explicitly set
# Complex file list: A file containing the list of complexes to process -i in parser

def parser_run_pdbprocess():
  parser = argparse.ArgumentParser(description="Run the automatic pdb processing to establish the FEater database")
  parser.add_argument("-i", "--input", default="", type=str, help="Input protein file list. ")
  parser.add_argument("-d", "--directory", default="", type=str, help="Output directory; New folders named by the residues will be built in this folder. ")
  parser.add_argument("-r", "--performance_report", default=0, type=int, help="Write performance report (typically named FEater_perf_report.html) from dask; Default: 0. ")

  # Output file settings
  parser.add_argument("-wp", "--write_pdb", default=1, type=int, help="Write the pdb file; Default: 1. ")
  parser.add_argument("-wm", "--write_mol2", default=1, type=int, help="Write the mol2 file; Default: 1. ")
  parser.add_argument("-ws", "--write_sdf", default=1, type=int, help="Write the sdf file; Default: 1. ")
  parser.add_argument("-wo", "--write_surf", default=1, type=int, help="Write the surf file; Default: 1. ")

  parser.add_argument("-prod", "--production_mode", default=0, type=int, help="Run the production mode; Only explicitly set to 1 to run the production mode, otherwise run the test mode; Default: 0.")

  # Dask worker settings
  parser.add_argument("-wn", "--worker_number", default=8, type=int, help="Number of workers; Default: 8. ")
  parser.add_argument("-tn", "--thread_per_worker", default=2, type=int, help="Number of threads per worker; Default: 2. ")

  # Debug mode only use single thread
  parser.add_argument("--debug", default=0, type=int, help="Debug mode")
  parser.add_argument("--debug_res", default="ALA", type=str, help="Debug mode: Focused residue to process")
  parser.add_argument("--debug_proteinnr", default=30, type=int, help="Debug mode: Number of proteins to process")
  args = parser.parse_args()

  camp_dir = os.environ.get("CAMPARI_DIR", "")
  camp_exe = os.environ.get("CAMPARI_EXE", "")

  if camp_dir == "" or camp_exe == "":
    raise ValueError("Please define the environment variables: CAMPARI_DIR, CAMPARI_EXE")
  if not os.path.exists(camp_dir):
    raise FileNotFoundError(f"Cannot find the campari directory: {camp_dir}")
  if not os.path.exists(camp_exe):
    raise FileNotFoundError(f"Cannot find the campari executable: {camp_exe}")
  if not os.path.exists(args.input):
    print(f"Cannot find the input file list: {args.input}")
    parser.print_help()
    exit(1)
  if not os.path.exists(args.directory):
    print(f"Cannot find the output directory: {args.directory}")
    parser.print_help()
    exit(1)

  return args

def run_pdbprocess():
  args = parser_run_pdbprocess()
  complex_filelist = args.input
  output_folder = args.directory

  if not os.path.exists(complex_filelist):
    raise FileNotFoundError(f"Cannot find the complex file list: {complex_filelist}")
  if not os.path.exists(output_folder):
    raise FileNotFoundError(f"Cannot find the output folder: {output_folder}")

  complex_files = read_protein_list(complex_filelist)
  print("Total number of complexes: ", len(complex_files))

  update_config(write_pdb=args.write_pdb,
                write_mol2=args.write_mol2,
                write_sdf=args.write_sdf,
                write_surf=args.write_surf,
                output_folder=args.directory)

  if int(args.debug):
    # Run the debug mode: Use only one thread without parallelization via dask
    print("Running the debug mode")
    print(f"Processing {args.debug_proteinnr} complexes for residue {args.debug_res}")
    choosen_files = np.random.choice(complex_files, args.debug_proteinnr, replace=False)
    choosen_res = str(args.debug_res)
    do_debug(choosen_files, choosen_res, SETTINGS)
  else:
    worker_num = args.worker_number
    thread_per_worker = args.thread_per_worker
    default_protein_nr = 30
    if int(args.production_mode):
      # Run the production mode: process all the complexes and all the residues and parallelize via dask
      print("Running the production mode: ")
      print(f"Processing all the complexes for each residue")
      do_prod(complex_files, RESIDUES_THREE_LETTER, SETTINGS, worker_num, thread_per_worker)
    else:
      # Run the test mode: process 30 complexes and all residues and parallelize via dask for sanity check
      # if --prod is not explicitly set
      print("Running the test mode: ")
      print(f"Processing {default_protein_nr} complexes for each residue")
      choosen_files = np.random.choice(complex_files, default_protein_nr, replace=False)
      do_prod(choosen_files, RESIDUES_THREE_LETTER, SETTINGS, worker_num, thread_per_worker)

RESIDUES_THREE_LETTER = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS',
                         'GLU', 'GLN', 'GLY', 'HIS', 'ILE',
                         'LEU', 'LYS', 'MET', 'PHE', 'PRO',
                         'SER', 'THR', 'TRP', 'TYR', 'VAL']

def write_makefile(outfolder):
  makefilepath = os.path.join(outfolder, "Makefile")
  if os.path.isdir(outfolder) and (not os.path.exists(makefilepath)):
    with open(makefilepath, "w") as f:
      f.write("clean: \n  rm -rf ./ALA  ./ARG  ./ASN  ./ASP  ./CYS  ./GLN  ./GLY  ./HIS  ./ILE  ./LYS  ./MET  ./PHE  ./PRO  ./THR  ./TRP  ./TYR  ./VAL ./LEU ./GLU ./SER\n")


if __name__ == "__main__":
  # Read the input file list
  complex_filelist = "/MieT5/BetaPose/data/complexes/complex_filelist.txt"
  complex_files = read_protein_list(complex_filelist)
  print("Total number of complexes: ", len(complex_files))
  output_settings = {
    "write_pdb": True,
    "write_mol2": True,
    "write_sdf": True,
    "write_surf": True,
    "output_folder": "/disk2b/yzhang/feater_test3"
  }
  write_makefile(output_settings["output_folder"])
  # update_config(**output_settings)

  if int(os.environ.get("FEATER_TEST", False)):
    choosen_files = np.random.choice(complex_files, 30, replace=False)
    do_debug(choosen_files, "ALA", output_settings)

  else:
    print("Running the production mode")
    choosen_files = np.random.choice(complex_files, 30, replace=False)
    choosed_res = np.random.choice(RESIDUES_THREE_LETTER, 10, replace=False)
    do_prod(choosen_files, choosed_res, output_settings,worker_num=16, thread_per_worker=1)

