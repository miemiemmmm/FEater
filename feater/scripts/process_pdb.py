import os, sys, time
import numpy as np

import pytraj as pt
import siesta
import feater

# Read the input file




def parallelize_traj(traj_list, focused_res):
  print(f"Processing the residue: {focused_res}")
  traj_list = [i for i in traj_list]
  st_this = time.perf_counter()

  for complex_idx, complex_file in enumerate(traj_list):
    print(f"Processing {complex_file} ({complex_idx + 1}/{len(traj_list)}, last complex took {(time.perf_counter() - st_this):.2f} seconds)")
    st_this = time.perf_counter()
    complex_i = feater.io.StructureProcessor()
    complex_i.focus_residue(focused_res)
    complex_i.read_structure(complex_file)
    complex_i.update_config(write_pdb=True,
                            write_mol2=True,
                            write_sdf=True,
                            write_surf=True,
                            output_folder="/tmp/test_single_res/")
    complex_i.reset()
    complex_i.process_residue_3letter(focused_res)




if __name__ == "__main__":
  import dask
  from dask.distributed import Client, performance_report, LocalCluster

  RESIDUES_THREE_LETTER = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS',
                           'GLU', 'GLN', 'GLY', 'HIS', 'ILE',
                           'LEU', 'LYS', 'MET', 'PHE', 'PRO',
                           'SER', 'THR', 'TRP', 'TYR', 'VAL']

  # Read the input file
  complex_folder = "/MieT5/BetaPose/data/complexes/"
  complex_file_list = "complex_filelist.txt"
  with open(os.path.join(complex_folder, complex_file_list), 'r') as f:
    complex_files = [i for i in f.read().strip("\n").split("\n") if i != ""]
  complex_files = [os.path.join(complex_folder, i) for i in complex_files]

  if int(os.environ.get("FEATER_TEST", False)):
    print("Running the debug mode")
    complex_files = complex_files[:100]
    parallelize_traj(complex_files, "THR")
    exit(0)

  print("Running the production mode")
  worker_num = 32
  thread_per_worker = 1
  found_PDB = complex_files
  split_groups = np.array_split(found_PDB, worker_num)
  cluster = LocalCluster(
    n_workers=worker_num,
    threads_per_worker=thread_per_worker,
    processes=True,
    memory_limit='6GB',
  )

  with Client(cluster) as client:
    with performance_report(filename="dask-report.html"):
      tasks = []
      c = 0
      for trajlist in split_groups:
        for residue in RESIDUES_THREE_LETTER:
          c += 1
          print(f"Task {c} | focused residue: {residue} | trajectories: {trajlist}")
          tasks.append(dask.delayed(parallelize_traj)(trajlist, residue))

      print(f"# Task set contains {len(tasks)} jobs; Jobs are generated and ready to run")
      st = time.perf_counter()
      futures = client.compute(tasks)
      results = client.gather(futures)
      [i.release() for i in futures]
      print(f"# Task finished, time elapsed: {(time.perf_counter() - st) / 60:.2f} min")

