import os, sys, argparse, time

import pytraj as pt
import numpy as np


def checkfiles(file_list:str) -> list:
  with open(file_list, 'r') as f:
    files = f.read().strip("\n").split('\n')
    for file in files:
      if not os.path.isfile(file):
        raise ValueError(f"File {file} does not exist.")
  return files


def make_netCDF(nc_name:str, coord_files:list, **kwargs):
  c = 0
  for file in coord_files:
    if c == 0:
      final_traj = pt.load(file)
      c += 1
      continue
    else:
      traji = pt.load(file)
      final_traj.append(traji)
      c += 1
    if ((c) % 1000 == 0) or (c == len(coord_files)):
      print(f"Processed {c} files")
    if c == len(coord_files):
      print(f"Final traj has {final_traj.n_frames} frames")
      if final_traj.n_frames != len(coord_files):
        print(f"Warning: Final traj has {final_traj.n_frames} frames, but there are {len(coord_files)} files", file=sys.stderr)
      final_traj.save(nc_name, overwrite=True)
      print(f"Done processing {c} files")


def parse_args():
  parser = argparse.ArgumentParser(description="Convert coordinate files to HDF5 format")
  parser.add_argument("-i", "--input", type=str, help="The input file containing a list of coordinate files")
  parser.add_argument("-o", "--output", type=str, help="The output HDF5 file")
  parser.add_argument("-f", "--force", type=int, default=0, help="Force overwrite the output file")
  args = parser.parse_args()

  if (args.input is None) or (not os.path.exists(args.input)):
    print("Fatal: Please specify the input file", file=sys.stderr)
    # print help
    parser.print_help()
    exit(1)

  if (args.output is None):
    print("Fatal: Please specify the output file", file=sys.stderr)
    parser.print_help()
    exit(1)

  if (os.path.exists(args.output)) and (args.force == 0):
    print(f"Fatal: Output file '{args.output}' exists. Use -f to force overwrite", file=sys.stderr)
    parser.print_help()
    exit(1)
  elif (os.path.exists(args.output)) and (args.force == 1):
    print(f"Removing the existing output file '{args.output}'")
    if os.path.exists(args.output):
      os.remove(args.output)
  return args


def console_interface():
  args = parse_args()
  files = checkfiles(args.input)
  print(f"Found {len(files)} files in the list")
  make_netCDF(args.output, files)


if __name__ == "__main__":
  console_interface()


