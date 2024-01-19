import time, argparse, sys, os, subprocess, contextlib, tempfile, shutil
import numpy as np
import open3d as o3d
import pytraj as pt
from hashlib import md5

from feater import io, dataloader
from siesta.scripts import view_obj


TMPDIR = os.path.abspath("./")

GEOM_DICT = {}

def molecule_to_o3d(struct_top, coords):
  """
  Convert a molecule to Open3D geometries.
  Args:
    struct_top: pytraj topology object
    coords: coordinates of the molecule
  """
  global GEOM_DICT
  atoms = list(struct_top.atoms)
  residues = list(struct_top.residues)

  # Add atoms as spheres: hash the atom (f"atom_{theatom.name}_{resname}_{atomtype}") to make it unique
  for idx, c in enumerate(coords):
    theatom = atoms[idx]
    resname = residues[theatom.resid].name
    atomtype = view_obj.getAtomNum(theatom.name, resname)
    char_string = f"atom_{theatom.name}_{resname}_{atomtype}"
    char_hash = md5(char_string.encode()).hexdigest()
    color = view_obj.element_color_map.get(atomtype, [0.5,0.5,0.5])
    sphere = view_obj.create_sphere(c, radius=0.5, color=color)
    sphere.compute_triangle_normals()
    GEOM_DICT[char_hash] = sphere

  # Add bonds as cylinders: hash the bond (f"bond_{n_i}_{n_j}") to make it unique
  for bond in list(struct_top.bonds):
    n_i, n_j = bond.indices
    pos_1 = coords[n_i]
    pos_2 = coords[n_j]
    bond_hash = md5(f"bond_{n_i}_{n_j}".encode()).hexdigest()
    if np.linalg.norm(pos_1 - pos_2) < 3:  # Simple condition to check if there is a bond
      cylinder = view_obj.create_cylinder(pos_1, pos_2, radius=0.15)
      GEOM_DICT[bond_hash] = cylinder


def update_molecule_geometry(struct_top, new_coords):
  """
  Update the global geometries of the molecule
  Args:
    struct_top: pytraj topology object
    new_coords: new coordinates of the molecule
  """
  global GEOM_DICT
  atoms = list(struct_top.atoms)
  residues = list(struct_top.residues)
  for idx, c in enumerate(new_coords):
    theatom = atoms[idx]
    resname = residues[theatom.resid].name
    atomtype = view_obj.getAtomNum(theatom.name, resname)
    char_string = f"atom_{theatom.name}_{resname}_{atomtype}"
    char_hash = md5(char_string.encode()).hexdigest()
    if char_hash in GEOM_DICT.keys():
      # print(f"Updating Atom {char_hash}")
      update_sphere_center(GEOM_DICT[char_hash], c)
      GEOM_DICT[char_hash].compute_vertex_normals()
    else:
      print(f"Warning: Not found Atom {char_hash}")

  for bond in list(struct_top.bonds):
    n_i, n_j = bond.indices
    pos_1 = new_coords[n_i]
    pos_2 = new_coords[n_j]
    bond_hash = md5(f"bond_{n_i}_{n_j}".encode()).hexdigest()
    if bond_hash in GEOM_DICT.keys():
      new_cylinder = view_obj.create_cylinder(pos_1, pos_2)
      # new_cylinder
      GEOM_DICT[bond_hash].vertices = new_cylinder.vertices
      GEOM_DICT[bond_hash].triangles = new_cylinder.triangles
      GEOM_DICT[bond_hash].compute_vertex_normals()
    else:
      print(f"Warning: Not found Bond {bond_hash}")


def update_sphere_center(sphere, new_center):
  """
  Update the center of a given sphere.
  Args:
    sphere: the sphere object
    new_center: the new center
  """
  transformation = np.eye(4)
  transformation[:3, 3] = np.array(new_center) - np.array(sphere.get_center())
  sphere.transform(transformation)


def read_h5_trajs(h5files, topfile, stride): 
  top = pt.load(topfile).top
  dataset = dataloader.CoordDataset(h5files, padding=False)
  frames = np.arange(0, len(dataset), stride)
  xyzs = np.zeros((len(frames), top.n_atoms, 3))
  
  for i, frame in enumerate(frames):
    data, _ = dataset[frame]
    data = data.numpy()
    xyzs[i] = data
  
  traj = pt.Trajectory(top=top, xyz=xyzs)
  return traj

def read_nc_traj(trajfiles, topfile, stride):
  thetraj = pt.iterload(trajfiles, top=topfile, stride=known_args.stride)
  return thetraj


@contextlib.contextmanager
def tempdir(make_movie=False):
  global TMPDIR
  if make_movie:
    tmpdir = tempfile.mkdtemp()
    TMPDIR = tmpdir
    print(f"Using temporary directory: {TMPDIR}")
    try:
      yield tmpdir
    finally:
      shutil.rmtree(tmpdir)
  else:
    yield None


def viewtraj_runner():
  global GEOM_DICT
  known_args, unknown_args = viewtraj_parser()
  fps = known_args.target_fps
  frametime = 1 / fps
  print("Visualization setings: ", known_args)

  # topfile = "/disk2b/yzhang/feater_test1/PRO/PRO_14_e632ce2b.pdb"
  topfile = known_args.topology
  trajfiles = []
  for trajfile in unknown_args:
    if not os.path.exists(trajfile):
      print(f"Fatal: The trajectory file '{trajfile}' does not exist", file=sys.stderr)
      exit(1)
    else:
      trajfiles.append(trajfile)

  print("The following trajectory files will be loaded: ")
  print("\n".join(trajfiles))
  if "h5" in trajfiles[0]:
    thetraj = read_h5_trajs(trajfiles, topfile, known_args.stride)
  elif "nc" in trajfiles[0] or "pdb" in trajfiles[0]:
    thetraj = pt.iterload(trajfiles, top=topfile, stride=known_args.stride)
  else: 
    print(f"Fatal: Unsupport trajectory file format", file=sys.stderr)
    exit(1)
  frames = thetraj.n_frames

  if known_args.alignment:
    selected_atoms = thetraj.top.select(known_args.alignment_mask)
    if len(selected_atoms) == 0:
      print(f"Fatal: The alignment mask '{known_args.alignment_mask}' is not valid", file=sys.stderr)
      exit(1)
    else:
      print(f"Aligning the trajectory with the following atoms: {selected_atoms}")
      thetraj.superpose(mask=known_args.alignment_mask)

  # Generate the viewer and add the model geometries
  molecule_to_o3d(thetraj.top, thetraj.xyz[0])  # Initialize the geometries (global variable GEOM_DICT)
  vis = o3d.visualization.Visualizer()
  vis.create_window(window_name="Trajectory viewer", width=1500, height=1000)
  for geom in GEOM_DICT.values():
    vis.add_geometry(geom)
  
  if known_args.render_json != "":
    vis.get_render_option().load_from_json(known_args.render_json)
  if known_args.camera_json != "":
    vis.get_view_control().convert_from_pinhole_camera_parameters(o3d.io.read_pinhole_camera_parameters(known_args.camera_json))

  vis.poll_events()
  vis.update_renderer()
  vis.run()              # Stall the program and allow the user to interact with the viewer

  with tempdir(make_movie=known_args.make_film):
    time_checker = time.perf_counter()
    for i in range(frames):
      if i % 100 == 0:
        print(f"Updating the geometry {i}/{frames}")
      update_molecule_geometry(thetraj.top, thetraj.xyz[i])
      for hash, geom in GEOM_DICT.items():
        vis.update_geometry(geom)
      vis.poll_events()
      vis.update_renderer()
      if known_args.make_film:
        vis.capture_screen_image(os.path.join(TMPDIR, f"frame_{i:03d}.png"))
      if (time.perf_counter() - time_checker) < frametime:
        # If the frame is rendered too fast, wait for a while
        time.sleep(frametime - (time.perf_counter() - time_checker))
      time_checker = time.perf_counter()

    # After rendering all the frames, synthesize the frames into a film
    if known_args.make_film:
      path_template = os.path.join(TMPDIR, "frame_%03d.png")
      command = f"ffmpeg -y -r 24 -f image2 -i {path_template} -vcodec libx264 -crf 25 -pix_fmt yuv420p output.mp4"
      print("Command for ffmpeg: ", command)
      ret = subprocess.call(command, shell=True)
      if ret != 0:
        print(f"Fatal: Failed to make the film", file=sys.stderr)
      else:
        print(f"Successfully made the film: output.mp4")
  vis.destroy_window()


def viewtraj_parser():
  parser = argparse.ArgumentParser(description="View the trajectory of a particle")
  parser.add_argument("-t", "--topology", type=str, help="The topology file")
  parser.add_argument("-fps", "--target_fps", type=int, default=27, help="The target FPS")
  parser.add_argument("-f", "--make_film", type=int, default=0, help="Make a film")
  parser.add_argument("-ac", "--automatic_center", type=int, default=0, help="Automatically center the particle")
  parser.add_argument("-a", "--alignment", default=0, type=int, help="Align the particle")
  parser.add_argument("-am", "--alignment_mask", type=str, default="@CA,C,N", help="The alignment mask")
  parser.add_argument("-s", "--stride", type=int, default=50, help="The stride of the trajectory")
  parser.add_argument("-r", "--render_json", type=str, default="", help="The camera settings")
  parser.add_argument("-c", "--camera_json", type=str, default="", help="The camera settings")

  known_args, unknown_args = parser.parse_known_args()

  if (known_args.topology is None) or (not os.path.exists(known_args.topology)):
    print("Fatal: Please specify the topology file", file=sys.stderr)
    parser.print_help()
    exit(1)
  if len(unknown_args) == 0:
    print("Fatal: Please specify the trajectory file", file=sys.stderr)
    parser.print_help()
    exit(1)
  return known_args, unknown_args


if __name__ == "__main__":
  viewtraj_runner()
  
  # python /MieT5/MyRepos/FEater/feater/scripts/view_traj.py -t /media/yzhang/MieT72/Data/feater_database_coord/Topology_ASN.pdb \
  # /media/yzhang/MieT72/Data/feater_database_coord/ValidationSet_ASN.h5 -a 1 -am @C,CA,N -f 1 



