import argparse

import numpy as np
import open3d as o3d

from feater import io


def get_coordi(hdf, index:int) -> np.ndarray:
  coord_sti = hdf["xyzr_starts"][index]
  coord_end = hdf["xyzr_ends"][index]
  coordi = hdf["xyzr"][coord_sti:coord_end]

  ret = []
  for i in range(coordi.shape[0]):
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=coordi[i,3]*1)
    sphere.translate(coordi[i,:3])
    sphere.paint_uniform_color([1,0,0])
    sphere.compute_vertex_normals()
    ret.append(sphere)
  return ret


def get_surfi(hdf, index:int) -> np.ndarray:
  vert_sti = hdf["vert_starts"][index]
  vert_end = hdf["vert_ends"][index]
  face_sti = hdf["face_starts"][index]
  face_end = hdf["face_ends"][index]
  vert = hdf["vertices"][vert_sti:vert_end]
  face = hdf["faces"][face_sti:face_end]
  surf = o3d.geometry.TriangleMesh()
  surf.vertices = o3d.utility.Vector3dVector(vert)
  surf.triangles = o3d.utility.Vector3iVector(face)
  surf.compute_vertex_normals()
  surf.paint_uniform_color([0.5, 0.5, 0.5])
  print(f"The surface has {len(vert)} vertices and {len(face)} faces.")
  return surf


def main_render(inputfile:str, index:int, args):
  with io.hdffile(inputfile, "r") as hdf:
    # Get the necessary geometries
    surf = get_surfi(hdf, index)

  vis = o3d.visualization.Visualizer()
  vis.create_window(window_name="Surface viewer", width=1400, height=1400)

  if args.wireframe:
    wire_frame = o3d.geometry.LineSet.create_from_triangle_mesh(surf)
    wire_frame.paint_uniform_color([0,0,1])
    vis.add_geometry(wire_frame)
  else:
    vis.add_geometry(surf)

  if args.balls:
    with io.hdffile(inputfile, "r") as hdf: 
      if "xyzr_starts" not in hdf.keys():
        print("Fatal: The input HDF5 file does not contain the atom XYZR (Not likely a molecular dataset).")
        exit(1)
      balls = get_coordi(hdf, index)
    for ball in balls:
      vis.add_geometry(ball)
  vis.run()
  vis.destroy_window()
  
  if args.save:
    # Save the object to a file
    final_obj = o3d.geometry.TriangleMesh()
    # Only add points 
    from matplotlib import colormaps
    cmap = colormaps.get_cmap("jet")
    vertnr = np.array(surf.vertices).shape[0]
    for idx, vert in enumerate(surf.vertices):
      if np.linalg.norm(vert) == 0:
        continue
      box = o3d.geometry.TriangleMesh.create_box(width=0.075, height=0.075, depth=0.075)
      box.translate(vert)
      color = cmap(idx/vertnr)[:3]
      box.paint_uniform_color(color)
      final_obj += box

    if args.balls:
      for ball in balls:
        ball.compute_vertex_normals()
        final_obj += ball
    o3d.io.write_triangle_mesh("/home/yzhang/Desktop/final_obj.ply", final_obj, write_ascii=True, write_vertex_normals=True, write_vertex_colors=True)


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("-f", "--fileinput", type=str, help="The input HDF5 file")
  parser.add_argument("-i", "--index", type=int, default=0, help="The index of the molecule to be viewed; Default: 0 ")
  parser.add_argument("-w", "--wireframe", type=int, default=1, help="Show the wireframe of the surface; Default: 1 ")
  parser.add_argument("-b", "--balls", type=int, default=0, help="Show the atoms as balls (xyz+radius); Default: 0 ")
  parser.add_argument("-s", "--save", type=int, default=0, help="Save the final object to a file; Default: 0.")
  args = parser.parse_args()
  return args


def console_interface():
  args = parse_args()
  print(args)
  main_render(args.fileinput, args.index, args)


if __name__ == "__main__":
  console_interface()

