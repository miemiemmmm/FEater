import os, argparse, sys, time

import numpy as np
from pytraj import load as ptload
from feater import RES2LAB, io, utils
import siesta


def mol_to_surf(molfile:str,spacing=0.35,smooth_step=1,slice_number=300,**kwargs):
  thexyzr = siesta.pdb_to_xyzr(molfile)
  verts, faces = siesta.xyzr_to_surf(thexyzr,
                                     grid_size=spacing,
                                     slice_number=slice_number,
                                     smooth_step=smooth_step)
  verts = np.array(verts, dtype=np.float32)
  faces = np.array(faces, dtype=np.int32)
  return verts, faces, thexyzr


# @profile
def make_hdf(hdf_name:str, coord_files:list, **kwargs):
  st = time.perf_counter()
  SurfConfig = {
    "grid_spacing": kwargs.get("grid_spacing", 0.35),
    "smooth_step": kwargs.get("smooth_step", 1),
    "slice_number": kwargs.get("slice_number", 300),
  }
  with io.hdffile(hdf_name, 'w') as f:
    # Surface generation parameters
    utils.add_data_to_hdf(f, "grid_spacing", np.array([SurfConfig["grid_spacing"]], dtype=np.float32), maxshape=[1],
                          dtype=np.float32)
    utils.add_data_to_hdf(f, "smooth_step", np.array([SurfConfig["smooth_step"]], dtype=np.int32), maxshape=[1],
                          dtype=np.int32)
    utils.add_data_to_hdf(f, "slice_number", np.array([SurfConfig["slice_number"]], dtype=np.int32), maxshape=[1],
                          dtype=np.int32)
    # Initialize the counters/pointers/esitmated sizes
    c = 0
    global_start_idx_xyzr = 0
    global_start_idx_vert = 0
    global_start_idx_face = 0
    output_interval = 1000
    estimate_coord_size = 1000
    estimate_vert_size = 4000
    estimate_face_size = int(estimate_vert_size * 2.5)

    # Initialize the buffers for different types of data
    label_buffer = np.full((output_interval), -1, dtype=np.int32)
    xyzr_st_buffer = np.zeros((output_interval), dtype=np.uint64)
    xyzr_ed_buffer = np.zeros((output_interval), dtype=np.uint64)
    vert_st_buffer = np.zeros((output_interval), dtype=np.uint64)
    vert_ed_buffer = np.zeros((output_interval), dtype=np.uint64)
    face_st_buffer = np.zeros((output_interval), dtype=np.uint64)
    face_ed_buffer = np.zeros((output_interval), dtype=np.uint64)

    pointer_xyzr = 0
    pointer_vert = 0
    pointer_face = 0
    xyzr_buffer = np.zeros((estimate_coord_size*output_interval, 4), dtype=np.float32)
    elems_buffer = np.zeros((estimate_coord_size*output_interval), dtype=np.int32)
    vertex_buffer = np.zeros((estimate_vert_size*output_interval, 3), dtype=np.float32)
    face_buffer = np.zeros((estimate_face_size*output_interval, 3), dtype=np.int32)

    f.draw_structure()
    for fidx, file in enumerate(coord_files):
      file = os.path.abspath(file)
      verts, faces, xyzr = mol_to_surf(file, spacing=SurfConfig["grid_spacing"], smooth_step=SurfConfig["smooth_step"], slice_number=SurfConfig["slice_number"])
      elems = np.array(ptload(file).top.mass).round().astype(np.int32)

      nr_verts = verts.shape[0]
      nr_faces = faces.shape[0]
      nr_xyzr = xyzr.shape[0]
      start_xyzri = global_start_idx_xyzr
      end_xyzri = start_xyzri + nr_xyzr
      start_verti = global_start_idx_vert
      end_verti = start_verti + nr_verts
      start_facei = global_start_idx_face
      end_facei = start_facei + nr_faces
      global_start_idx_xyzr += nr_xyzr
      global_start_idx_vert += nr_verts
      global_start_idx_face += nr_faces

      # Self adjust the estimated length of each element buffer size
      if nr_xyzr > estimate_coord_size:
        estimate_coord_size = int(nr_xyzr * 1.25)
      if nr_verts > estimate_vert_size:
        estimate_vert_size = int(nr_verts * 1.25)
      if nr_faces > estimate_face_size:
        estimate_face_size = int(nr_faces * 1.25)

      # Get the label of the residue
      restype = os.path.basename(file)[:3]
      if restype not in RES2LAB.keys():
        raise ValueError(f"Unknown residue type: {restype}")
      labeli = RES2LAB[restype]

      current_idx = fidx % output_interval
      label_buffer[current_idx] = labeli
      vert_st_buffer[current_idx] = start_verti
      vert_ed_buffer[current_idx] = end_verti
      face_st_buffer[current_idx] = start_facei
      face_ed_buffer[current_idx] = end_facei
      xyzr_st_buffer[current_idx] = start_xyzri
      xyzr_ed_buffer[current_idx] = end_xyzri

      # Mask the buffers based on the pointer
      if pointer_xyzr + nr_xyzr > xyzr_buffer.shape[0]:
        xyzr_buffer = np.zeros((estimate_coord_size * output_interval, 4), dtype=np.float32)
        xyzr_buffer[:pointer_xyzr] = xyzr_buffer[:pointer_xyzr]
        elems_buffer = np.zeros((estimate_coord_size * output_interval), dtype=np.int32)
        elems_buffer[:pointer_xyzr] = elems_buffer[:pointer_xyzr]
      if pointer_vert + nr_verts > vertex_buffer.shape[0]:
        vertex_buffer = np.zeros((estimate_vert_size * output_interval, 3), dtype=np.float32)
        vertex_buffer[:pointer_vert] = vertex_buffer[:pointer_vert]
      if pointer_face + nr_faces > face_buffer.shape[0]:
        face_buffer = np.zeros((estimate_face_size * output_interval, 3), dtype=np.int32)
        face_buffer[:pointer_face] = face_buffer[:pointer_face]
      xyzr_buffer[pointer_xyzr:pointer_xyzr+nr_xyzr, :] = xyzr
      elems_buffer[pointer_xyzr:pointer_xyzr+nr_xyzr] = elems
      vertex_buffer[pointer_vert:pointer_vert+nr_verts, :] = verts
      face_buffer[pointer_face:pointer_face+nr_faces, :] = faces
      pointer_xyzr += nr_xyzr
      pointer_vert += nr_verts
      pointer_face += nr_faces

      c += 1
      if (c % output_interval == 0) or (c == len(coord_files)):
        print(f"Processed {c} files, Time elapsed: {time.perf_counter() - st:.2f} seconds")
        label_count = np.count_nonzero(label_buffer >= 0)
        if label_count != len(label_buffer):
          label_buffer = label_buffer[:label_count]
          vert_st_buffer = vert_st_buffer[:label_count]
          vert_ed_buffer = vert_ed_buffer[:label_count]
          face_st_buffer = face_st_buffer[:label_count]
          face_ed_buffer = face_ed_buffer[:label_count]
          xyzr_st_buffer = xyzr_st_buffer[:label_count]
          xyzr_ed_buffer = xyzr_ed_buffer[:label_count]
        xyzr_buffer = xyzr_buffer[:pointer_xyzr]
        elems_buffer = elems_buffer[:pointer_xyzr]
        vertex_buffer = vertex_buffer[:pointer_vert]
        face_buffer = face_buffer[:pointer_face]

        utils.add_data_to_hdf(f, "xyzr", xyzr_buffer, dtype=np.float32, maxshape=[None, 4], compression="gzip", compression_opts=1)
        utils.add_data_to_hdf(f, "vertices", vertex_buffer, dtype=np.float32, maxshape=[None, 3], compression="gzip", compression_opts=1)
        utils.add_data_to_hdf(f, "faces", face_buffer, dtype=np.int32, maxshape=[None, 3], compression="gzip", compression_opts=1)
        utils.add_data_to_hdf(f, "label", label_buffer, dtype=np.int32, maxshape=[None])
        utils.add_data_to_hdf(f, "coord_starts", xyzr_st_buffer, dtype=np.uint64, maxshape=[None])
        utils.add_data_to_hdf(f, "coord_ends", xyzr_ed_buffer, dtype=np.uint64, maxshape=[None])
        utils.add_data_to_hdf(f, "vert_starts", vert_st_buffer, dtype=np.uint64, maxshape=[None], compression="gzip", compression_opts=1)
        utils.add_data_to_hdf(f, "vert_ends", vert_ed_buffer, dtype=np.uint64, maxshape=[None], compression="gzip", compression_opts=1)
        utils.add_data_to_hdf(f, "face_starts", face_st_buffer, dtype=np.uint64, maxshape=[None], compression="gzip", compression_opts=1)
        utils.add_data_to_hdf(f, "face_ends", face_ed_buffer, dtype=np.uint64, maxshape=[None], compression="gzip", compression_opts=1)
        utils.add_data_to_hdf(f, "element_mass", elems_buffer, dtype=np.int32, maxshape=[None])

        # Reset the buffers and buffer pointers
        label_buffer = np.full((output_interval), -1, dtype=np.int32)
        xyzr_st_buffer = np.zeros((output_interval), dtype=np.uint64)
        xyzr_ed_buffer = np.zeros((output_interval), dtype=np.uint64)
        vert_st_buffer = np.zeros((output_interval), dtype=np.uint64)
        vert_ed_buffer = np.zeros((output_interval), dtype=np.uint64)
        face_st_buffer = np.zeros((output_interval), dtype=np.uint64)
        face_ed_buffer = np.zeros((output_interval), dtype=np.uint64)

        xyzr_buffer = np.zeros((estimate_coord_size*output_interval, 4), dtype=np.float32)
        elems_buffer = np.zeros((estimate_coord_size*output_interval), dtype=np.int32)
        vertex_buffer = np.zeros((estimate_vert_size*output_interval, 3), dtype=np.float32)
        face_buffer = np.zeros((estimate_face_size*output_interval, 3), dtype=np.int32)
        pointer_xyzr = 0
        pointer_vert = 0
        pointer_face = 0
        st = time.perf_counter()
    f.draw_structure()
    print(f"Each entry has {len(f['xyzr'])/len(f['label']):.2f} atoms, {len(f['vertices'])/len(f['label']):.2f} vertices, {len(f['faces'])/len(f['label']):.2f} faces")


def parse_args():
  parser = argparse.ArgumentParser(description="Make HDF5")
  parser.add_argument("-i", "--input", type=str, help="The file writes all of the absolute path of coordinate files")
  parser.add_argument("-o", "--output", type=str, help="The output HDF5 file")
  args = parser.parse_args()
  return args


def console_interface():
  args = parse_args()
  files = utils.checkfiles(args.input)
  print(f"Found {len(files)} files in the list")
  make_hdf(args.output, files)


if __name__ == "__main__":
  console_interface()
  # filelists = "/media/yzhang/MieT72/Data/feater_database/ValidationSet_ALA.txt%/media/yzhang/MieT72/Data/feater_database/ValidationSet_ARG.txt%/media/yzhang/MieT72/Data/feater_database/ValidationSet_ASN.txt%/media/yzhang/MieT72/Data/feater_database/ValidationSet_ASP.txt%/media/yzhang/MieT72/Data/feater_database/ValidationSet_CYS.txt%/media/yzhang/MieT72/Data/feater_database/ValidationSet_GLN.txt%/media/yzhang/MieT72/Data/feater_database/ValidationSet_GLU.txt%/media/yzhang/MieT72/Data/feater_database/ValidationSet_GLY.txt%/media/yzhang/MieT72/Data/feater_database/ValidationSet_HIS.txt%/media/yzhang/MieT72/Data/feater_database/ValidationSet_ILE.txt%/media/yzhang/MieT72/Data/feater_database/ValidationSet_LEU.txt%/media/yzhang/MieT72/Data/feater_database/ValidationSet_LYS.txt%/media/yzhang/MieT72/Data/feater_database/ValidationSet_MET.txt%/media/yzhang/MieT72/Data/feater_database/ValidationSet_PHE.txt%/media/yzhang/MieT72/Data/feater_database/ValidationSet_PRO.txt%/media/yzhang/MieT72/Data/feater_database/ValidationSet_SER.txt%/media/yzhang/MieT72/Data/feater_database/ValidationSet_THR.txt%/media/yzhang/MieT72/Data/feater_database/ValidationSet_TRP.txt%/media/yzhang/MieT72/Data/feater_database/ValidationSet_TYR.txt%/media/yzhang/MieT72/Data/feater_database/ValidationSet_VAL.txt%/media/yzhang/MieT72/Data/feater_database/TestSet_ALA.txt%/media/yzhang/MieT72/Data/feater_database/TestSet_ARG.txt%/media/yzhang/MieT72/Data/feater_database/TestSet_ASN.txt%/media/yzhang/MieT72/Data/feater_database/TestSet_ASP.txt%/media/yzhang/MieT72/Data/feater_database/TestSet_CYS.txt%/media/yzhang/MieT72/Data/feater_database/TestSet_GLN.txt%/media/yzhang/MieT72/Data/feater_database/TestSet_GLU.txt%/media/yzhang/MieT72/Data/feater_database/TestSet_GLY.txt%/media/yzhang/MieT72/Data/feater_database/TestSet_HIS.txt%/media/yzhang/MieT72/Data/feater_database/TestSet_ILE.txt%/media/yzhang/MieT72/Data/feater_database/TestSet_LEU.txt%/media/yzhang/MieT72/Data/feater_database/TestSet_LYS.txt%/media/yzhang/MieT72/Data/feater_database/TestSet_MET.txt%/media/yzhang/MieT72/Data/feater_database/TestSet_PHE.txt%/media/yzhang/MieT72/Data/feater_database/TestSet_PRO.txt%/media/yzhang/MieT72/Data/feater_database/TestSet_SER.txt%/media/yzhang/MieT72/Data/feater_database/TestSet_THR.txt%/media/yzhang/MieT72/Data/feater_database/TestSet_TRP.txt%/media/yzhang/MieT72/Data/feater_database/TestSet_TYR.txt%/media/yzhang/MieT72/Data/feater_database/TestSet_VAL.txt%/media/yzhang/MieT72/Data/feater_database/TrainingSet_ALA.txt%/media/yzhang/MieT72/Data/feater_database/TrainingSet_ARG.txt%/media/yzhang/MieT72/Data/feater_database/TrainingSet_ASN.txt%/media/yzhang/MieT72/Data/feater_database/TrainingSet_ASP.txt%/media/yzhang/MieT72/Data/feater_database/TrainingSet_CYS.txt%/media/yzhang/MieT72/Data/feater_database/TrainingSet_GLN.txt%/media/yzhang/MieT72/Data/feater_database/TrainingSet_GLU.txt%/media/yzhang/MieT72/Data/feater_database/TrainingSet_GLY.txt%/media/yzhang/MieT72/Data/feater_database/TrainingSet_HIS.txt%/media/yzhang/MieT72/Data/feater_database/TrainingSet_ILE.txt%/media/yzhang/MieT72/Data/feater_database/TrainingSet_LEU.txt%/media/yzhang/MieT72/Data/feater_database/TrainingSet_LYS.txt%/media/yzhang/MieT72/Data/feater_database/TrainingSet_MET.txt%/media/yzhang/MieT72/Data/feater_database/TrainingSet_PHE.txt%/media/yzhang/MieT72/Data/feater_database/TrainingSet_PRO.txt%/media/yzhang/MieT72/Data/feater_database/TrainingSet_SER.txt%/media/yzhang/MieT72/Data/feater_database/TrainingSet_THR.txt%/media/yzhang/MieT72/Data/feater_database/TrainingSet_TRP.txt%/media/yzhang/MieT72/Data/feater_database/TrainingSet_TYR.txt%/media/yzhang/MieT72/Data/feater_database/TrainingSet_VAL.txt"
  # listfiles = filelists.strip("%").split("%")
  # print(f"Processing {len(listfiles)} list files")
  #
  # basepath = "/media/yzhang/MieT72/Data/feater_database"
  # outputdir = "/media/yzhang/MieT72/Data/feater_database_surf"
  #
  # for listfile in listfiles[50:51]:
  #   resname = os.path.basename(listfile).split(".")[0].split("_")[1]
  #   _filename = os.path.basename(listfile).split(".")[0]
  #   outfile = os.path.join(outputdir, f"{_filename}.h5")
  #   files = utils.checkfiles(listfile, basepath=basepath)
  #   print(f"Found {len(files)} files in the {listfile}, will write to {outfile}")
  #   make_hdf(outfile, files)



