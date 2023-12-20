import os, time, random, argparse

import numpy as np
import open3d as o3d

from feater import io
from siesta.scripts import view_obj


def get_coordi(hdf, index:int) -> np.ndarray:
  coord_sti = hdf["coord_start"][index]
  coord_end = hdf["coord_end"][index]
  coordi = hdf["xyzr"][coord_sti:coord_end]

  ret = []
  for i in range(coordi.shape[0]):
    # print(i, coordi[i,:3], coordi[i,3])
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=coordi[i,3]*1)
    sphere.translate(coordi[i,:3])
    sphere.paint_uniform_color([1,0,0])
    sphere.compute_vertex_normals()
    ret.append(sphere)
  return ret

def get_surfi(hdf, index:int) -> np.ndarray:
  vert_sti = hdf["vert_start"][index]
  vert_end = hdf["vert_end"][index]
  face_sti = hdf["face_start"][index]
  face_end = hdf["face_end"][index]
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
    hdf.draw_structure()
    # Get the necessary geometries
    surf = get_surfi(hdf, index)
    balls = get_coordi(hdf, index)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Surface viewer", width=1400, height=1400)

    if args.wireframe:
      wire_frame = o3d.geometry.LineSet.create_from_triangle_mesh(surf)
      wire_frame.paint_uniform_color([0,0,1])
      vis.add_geometry(wire_frame)
    else:
      vis.add_geometry(surf)

    if args.balls:
      for ball in balls:
        vis.add_geometry(ball)
    vis.run()
    vis.destroy_window()


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("-f", "--fileinput", type=str, help="The input HDF5 file")
  parser.add_argument("-i", "--index", type=int, default=0, help="The index of the molecule to be viewed, default: 0 (no)")
  parser.add_argument("-w", "--wireframe", type=int, default=1, help="Show the wireframe of the surface, default: 1 (yes)")
  parser.add_argument("-b", "--balls", type=int, default=1, help="Show the atoms as balls (xyz+radius), default: 1 (yes)")
  args = parser.parse_args()
  return args


def console_interface():
  args = parse_args()
  print(args)
  main_render(args.fileinput, args.index, args)


if __name__ == "__main__":
  console_interface()

  # main_render("test.h5", 3400, None)
  # main_render("test.h5", 3500, None)

