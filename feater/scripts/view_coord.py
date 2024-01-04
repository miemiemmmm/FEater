import os, time, random, argparse, json

import numpy as np
import open3d as o3d

from matplotlib import colormaps
from feater import io
from siesta.scripts import view_obj


element_color_map = {
  # BASIC ELEMENTS
  1:  [1, 1, 1],             # Hydrogen
  12: [0.78, 0.78, 0.78],    # Carbon
  14: [0, 0, 1],             # Nitrogen
  16: [1, 0, 0],             # Oxygen
  32: [1.0 , 0.78, 0.20],    # Sulfur
  31: [1.0 , 0.64, 0],       # Phosphorus

  # METALS
  23: [0, 0, 1.0],           # Sodium
  24: [0.16, 0.5, 0.16],     # Magnesium
  39: [0, 0.5, 1],           # Potassium
  40: [0.5, 0.5, 0.5],       # Calcium
  56: [1 , 0.64, 0],         # Iron
  55: [0.16, 0.5, 0.16],     # Manganese
  64: [0.8, 0.4, 0.1],       # Copper
  65: [0.64, 0.16, 0.16],    # Zinc

  # Halogens
  17: [0, 1, 0],             # Chlorine
  35: [0.64, 0.16, 0.16],    # Bromine
  53: [0.48, 0.31, 0.62],    # Iodine

  # UNKNOWNS
  "UNK": [0.5, 0.5, 0.5],
  "U": [0.5, 0.5, 0.5],
}


KEY_COORD = "coordinates"
KEY_COORD_START_MAP = "coord_starts"
KEY_COORD_END_MAP = "coord_ends"
KEY_COLOR_HINT = "elements"


def get_coordi(hdf, index:int) -> np.ndarray:
  st = hdf[KEY_COORD_START_MAP][index]
  ed = hdf[KEY_COORD_END_MAP][index]
  coord = hdf[KEY_COORD][st:ed]
  return np.asarray(coord, dtype=np.float64)


def get_elemi(hdf, index:int) -> np.ndarray:
  st = hdf[KEY_COORD_START_MAP][index]
  ed = hdf[KEY_COORD_END_MAP][index]
  elemi = hdf[KEY_COLOR_HINT][st:ed]
  retcolor = np.zeros((elemi.shape[0], 3), dtype=np.float32)
  for idx in range(len(elemi)):
    mass = elemi[idx]
    if mass not in element_color_map:
      print(f"Warning: Element {mass} is not in the element color map")
      retcolor[idx] = element_color_map["UNK"]
    else:
      retcolor[idx] = element_color_map[mass]
  return retcolor


def get_geo_coordi(coordi) -> list:
  ret = []
  for i in range(coordi.shape[0]):
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.5)
    sphere.translate(coordi[i])
    sphere.paint_uniform_color([1,0,0])
    ret.append(sphere)
  return ret


def add_bounding_box(dims):  # TODO: Already put this to the SiESTA
  boxpoints = np.array([
    [0,0,0],
    [dims[0],0,0],
    [0, dims[1], 0],
    [0,0,dims[2]],
    [dims[0], dims[1], 0],
    [dims[0], 0, dims[2]],
    [0, dims[1], dims[2]],
    [dims[0], dims[1], dims[2]],
  ])
  lines = [
    [0,1], [0,2], [0,3], [1,4],
    [1,5], [2,4], [2,6], [3,5],
    [3,6], [4,7], [5,7], [6,7],
  ]
  ret = []
  for line in lines:
    cylinder = view_obj.create_cylinder(boxpoints[line[0]], boxpoints[line[1]], radius=0.1, color=[0,0,1])
    ret.append(cylinder)
  print(len(ret))
  for geo in ret:
    print(geo)
    geo.compute_vertex_normals()
  return ret


def main_render(inputfile:str, index:int, args):
  with io.hdffile(inputfile, "r") as hdf:
    hdf.draw_structure()

    if KEY_COORD not in hdf:
      raise ValueError("Cannot find the coordinates in the HDF file")
    elif KEY_COORD_START_MAP not in hdf or KEY_COORD_END_MAP not in hdf:
      raise ValueError("Cannot find the coordinate start/end map in the HDF file")
    if index >= hdf["label"].shape[0]:
      raise ValueError(f"Index {index} is out of range")
    
    coordi = get_coordi(hdf, index) 
    coord_cog = np.mean(coordi, axis=0)

    # Get the bounding box as the dims and adding a padding of 1 on each side
    dims = np.max(coordi, axis=0) - np.min(coordi, axis=0) + 2
    diff = coord_cog - (dims/2)
    colors = get_elemi(hdf, index)
    

  # Main rendering functions
  vis = o3d.visualization.Visualizer()
  vis.create_window(window_name="HDF Coordinate viewer", width=1500, height=1000)

  if args.top is not None:
    if not os.path.exists(args.top):
      raise ValueError(f"Topology file {args.top} does not exist")
    from pytraj import load, Trajectory
    traj = Trajectory(top=load(args.top).top, xyz=np.array([coordi - diff]))
    geoms_coord = view_obj.traj_to_o3d(traj)
    for geom in geoms_coord:
      vis.add_geometry(geom)
    
    
  else: 
    geoms_coord = get_geo_coordi(coordi - diff)
    for geo, color in zip(geoms_coord, colors):
      geo.paint_uniform_color(color)
      geo.compute_vertex_normals()
      vis.add_geometry(geo)

  # Add the bounding box
  if args.box:
    geoms_box = view_obj.create_bounding_box(dims)
    for geo in geoms_box:
      vis.add_geometry(geo)

  # Mark the center of the voxel
  if args.markcenter:
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1)
    sphere.translate(np.asarray(dims)/2)
    sphere.paint_uniform_color([0, 1, 0])
    vis.add_geometry(sphere)

  if args.render_json != "":
    vis.get_render_option().load_from_json(args.render_json)
  vis.poll_events()
  vis.update_renderer()
  if args.stuck:
    vis.run()
  else:
    time.sleep(0.1)
  out_prefix = os.path.basename(inputfile).split(".")[0]+f"{index:04d}"
  print(f"Saving the image to {out_prefix}.png")
  vis.capture_screen_image(f"{out_prefix}.png", True)
  vis.capture_depth_image(f"{out_prefix}_depth.png", True)
  vis.destroy_window()

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("-f", "--fileinput", type=str, help="The input HDF5 file")
  parser.add_argument("-i", "--index", type=int, default=0, help="The index of the molecule to be viewed")
  parser.add_argument("-b", "--box", type=int, default=1, help="Add the bounding box. Default: 1")
  parser.add_argument("-m", "--markcenter", default=1, type=int, help="Mark the center of the voxel (Marked by a green sphere). Default: 1")
  parser.add_argument("-r", "--render_json", type=str, default="", help="The camera settings")
  parser.add_argument("--top", type=str, help="The topology file (optional)")
  parser.add_argument("--stuck", type=int, default=1, help="Stuck the viewer. Default: 0")
  args = parser.parse_args()
  if args.fileinput is None:
    raise ValueError("Input file is not specified")
  if not os.path.exists(args.fileinput):
    raise ValueError(f"Input file {args.fileinput} does not exist")
  return args


def console_interface():
  args = parse_args()
  print(json.dumps(vars(args), indent=2))
  main_render(args.fileinput, args.index, args)

if __name__ == "__main__":
  console_interface()


# python /MieT5/MyRepos/FEater/feater/scripts/view_coord.py -f /media/yzhang/MieT72/Data/feater_database_coord/ValidationSet_${res}.h5 --top /media/yzhang/MieT72/Data/feater_database_coord/Topology_${res}.pdb -c ScreenCamera.json -i ${i} -m 0 -b 0
# python /MieT5/MyRepos/FEater/feater/scripts/view_coord.py -f /media/yzhang/MieT72/Data/feater_database_coord/ValidationSet_ASN.h5 -i 250 -m 0 --top /media/yzhang/MieT72/Data/feater_database_coord/Topology_ASN.pdb  -b 0