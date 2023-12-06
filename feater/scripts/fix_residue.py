import os, sys, copy, argparse, tempfile
import subprocess

CAMP_DIR = os.environ.get("CAMPARI_DIR", "/software/campari")
CAMP_EXE = os.environ.get("CAMPARI_EXE", "/software/campari/campari")
BASENAME = "resi_fixer"

TEMPLATE_KEY = {
  "PARAMETERS": os.path.abspath(os.path.join(CAMP_DIR, "params", "charmm36.prm")),
  "FMCSC_CMAPDIR": os.path.abspath(os.path.join(CAMP_DIR, "data")),
  "FMCSC_SIZE": 120,
  "FMCSC_SHAPE": 2,
  "FMCSC_BOUNDARY": 3,
  "FMCSC_SEQFILE": "test.seq",
  "FMCSC_PDBFILE": "test.pdb",
  "FMCSC_RANDOMIZE": 1,
  "FMCSC_RANDOMATTS": 1000,
  "FMCSC_RANDOMTHRESH": 3.0,
  "FMCSC_NRSTEPS": 1,
  "FMCSC_DYNAMICS": 2,
  "FMCSC_EQUIL": 100,
  "FMCSC_PDB_W_CONV": 4,
  "FMCSC_PDB_NUCMODE": 2,
  "FMCSC_BASENAME": BASENAME,
  "FMCSC_PDB_AUXINFO": 1,
  "FMCSC_UNSAFE": 1,
  "FMCSC_SYBYLLJMAP": os.path.abspath(os.path.join(CAMP_DIR, "params","abs4.2.ljmap"))
}


_debug = int(os.environ.get("FEATER_DEBUG", False))

if _debug:
  print("Default keywords are: ", TEMPLATE_KEY, file=sys.stderr)

def gen_key(keys:dict, key_file:str):
  new_key_dict = copy.deepcopy(TEMPLATE_KEY)
  for key in keys:
    if key in new_key_dict:
      new_key_dict[key] = keys[key]
  with open(key_file, "w") as f:
    for key in new_key_dict:
      f.write(f"{key} {new_key_dict[key]}\n")
  if os.path.exists(key_file):
    if _debug:
      print("Key file successfully generated", file=sys.stderr)
  else:
    raise FileNotFoundError(f"Cannot find the key file: {key_file}")


def fix_residue(target_dir:str, pdb_file:str="", seq_file:str="", key_file:str="", v=0) -> dict:
  # Check the folder and the input files (if provided)
  if not os.path.exists(target_dir):
    raise FileNotFoundError(f"Cannot find the target directory: {target_dir}")
  if len(pdb_file) > 0 and (not os.path.exists(pdb_file)):
    raise FileNotFoundError(f"Cannot find the pdb file: {pdb_file}")

  keyword_patch = {}
  campari_exe = os.environ.get("CAMPARI_EXE", "/software/campari/campari")
  if not os.path.exists(campari_exe):
    print(f"Cannot find the campari executable at {campari_exe}", file=sys.stderr)
    exit(1)

  campari_dir = os.environ.get("CAMPARI_DIR", "/software/campari")
  if not os.path.exists(campari_dir):
    print(f"Cannot find the campari directory at {campari_dir}", file=sys.stderr)
    exit(1)
  else:
    keyword_patch["PARAMETERS"] = os.path.abspath(os.path.join(campari_dir, "params", "charmm36.prm"))
    keyword_patch["FMCSC_CMAPDIR"] = os.path.abspath(os.path.join(campari_dir, "data"))
    keyword_patch["FMCSC_SYBYLLJMAP"] = os.path.abspath(os.path.join(campari_dir, "params","abs4.2.ljmap"))

  if len(seq_file) > 0:
    keyword_patch["FMCSC_SEQFILE"] = seq_file
  else:
    seq_file = os.path.join(target_dir, "test.seq")
    if not os.path.exists(seq_file):
      raise FileNotFoundError(f"Cannot find the default seq file: {seq_file}")
    else:
      keyword_patch["FMCSC_SEQFILE"] = seq_file

  if len(pdb_file) > 0:
    keyword_patch["FMCSC_PDBFILE"] = pdb_file
  else:
    pdb_file = os.path.join(target_dir, "test.pdb")
    if not os.path.exists(pdb_file):
      raise FileNotFoundError(f"Cannot find the default pdb file: {pdb_file}")
    else:
      keyword_patch["FMCSC_PDBFILE"] = pdb_file

  if len(key_file) > 0:
    gen_key(keyword_patch, key_file)
  else:
    key_file = os.path.join(target_dir, "test.key")
    gen_key(keyword_patch, key_file)

  cmd = f"cd {target_dir} && " + " ".join([campari_exe, "-k", key_file])

  if _debug or v:
    print("Working on the directory: ", os.path.abspath(os.path.curdir), file=sys.stderr)
    print("Campari residue fixer command is :", cmd, file=sys.stderr)

  try:
    if _debug or v:
      subprocess.run(cmd, shell=True)
    else:
      subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
  except subprocess.CalledProcessError as e:
    print("Residue fixer failed", file=sys.stderr)
    return {}

  expected_outpdb_name = os.path.join(target_dir, f"{BASENAME}_END.pdb")
  expected_outmol2_name = os.path.join(target_dir, f"{BASENAME}_END.mol2")
  if os.path.exists(expected_outpdb_name) and os.path.exists(expected_outmol2_name):
    with open(expected_outpdb_name, "r") as f:
      output_pdbstr = f.read()
    with open(os.path.join(target_dir, f"{BASENAME}_END.mol2"), "r") as f:
      output_mol2str = f.read()
    with open(pdb_file, "r") as f:
      input_pdbstr = f.read()
    if _debug or v:
      print("Residue fixer succeeded", file=sys.stderr)
      print("Output pdb string is: ", output_pdbstr, file=sys.stderr)
    resultdict = {
      "result_pdb": output_pdbstr,
      "result_mol2": output_mol2str,
      "input_pdb": input_pdbstr
    }
    return resultdict
  else:
    print("Residue fixer failed", file=sys.stderr)
    return {}


def parse_resfixer():
  parser = argparse.ArgumentParser(description="Fix the residue")
  parser.add_argument("-i", "--input", default="", type=str, help="Input pdb file")
  parser.add_argument("-s", "--seq", default="", type=str, help="Input sequence file in according to the campari format")
  parser.add_argument("-o", "--output", default="", type=str, help="Write the output pdb string to this file. If not defined, print to stdout")
  args = parser.parse_args()
  if (not args.input) or (not args.seq):
    parser.print_help()
    exit(1)
  if not os.path.exists(args.input):
    raise FileNotFoundError(f"Cannot find the input pdb file: {args.input}")
  if not os.path.exists(args.seq):
    raise FileNotFoundError(f"Cannot find the input seq file: {args.seq}")
  return args


def run_resfixer():
  args = parse_resfixer()
  if _debug:
    print("Check the arguments: ", args, file=sys.stderr)

  with tempfile.TemporaryDirectory() as tmpdir:
    file_input = os.path.join(tmpdir, "input.pdb")
    file_seq = os.path.join(tmpdir, "input.seq")
    file_key = os.path.join(tmpdir, "input.key")
    subprocess.check_call(f"cp {args.input} {file_input}", shell=True)
    subprocess.check_call(f"cp {args.seq} {file_seq}", shell=True)
    ret_str = fix_residue(tmpdir, pdb_file=file_input, seq_file=file_seq, key_file=file_key)
  if len(ret_str) == 0:
    print("Residue fixer failed", file=sys.stderr)
    exit(1)
  else:
    print("Residue fixer succeeded", file=sys.stderr)
    if len(args.output) > 0:
      with open(args.output, "w") as f:
        f.write(ret_str["result_pdb"])
    else:
      print(ret_str["result_pdb"])


if __name__ == "__main__":
  with open("/tmp/test.seq", "w") as f:
    f.write("HID_N_C\nEND\n")
  subprocess.check_call("rm -f *END*", shell=True)
  ret_str = fix_residue("/tmp")
  if len(ret_str) > 0:
    print("Residue fixer succeeded")
    print("Output pdb string is: \n", "\n".join(ret_str.split("\n")[:5]))
  else:
    print("Residue fixer failed")




