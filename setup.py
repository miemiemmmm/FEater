from setuptools import setup, find_packages
from setuptools.command.install import install
import sys, os, subprocess, shutil


class install_FEater(install):
  def run(self):
    # Install FEater
    # subprocess.call(["pip", "install", "feater"])
    subprocess.check_call(["make", "compile"])

    if os.path.isfile("./src/voxelize.so"):
      if os.path.isdir("./build/lib/feater"):
        shutil.copy2("./src/voxelize.so", "./build/lib/feater/voxelize.so")
      else:
        # Is this necessary?
        shutil.copy2("./src/voxelize.so", "./feater/voxelize.so")
    cwd = os.getcwd()
    subprocess.call(["find"])
    # shutil.copy2()
    print("######################################")

    install.run(self)



setup(
  cmdclass={"install": install_FEater},
  packages=find_packages(),
  zip_safe=False,
)