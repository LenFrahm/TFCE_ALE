import os
import sys
from mpi4py import MPI
from os.path import isfile, isdir
import pandas as pd
import numpy as np
import h5py
from utils.template import sample_space, affine
from utils.kernel import kernel_calc
from utils.compute import compute_ma, compute_hx, compute_hx_conv, compute_ale, compute_z, compute_tfce
from timeit import default_timer as timer


def ale(num_studies, num_true, rep):
  with h5py.File(f"/p/scratch/cinm-7/pyALE_TFCE/inputs/simulation_data_more_offset.hdf5", "r") as sim_data:
      sample_sizes = sim_data[f"{num_studies}/{num_true}/{rep}/sample_sizes"][:]
      foci = []
      for key in sim_data[f"{num_studies}/{num_true}/{rep}/foci"].keys():
          foci.append(sim_data[f"{num_studies}/{num_true}/{rep}/foci"][key][:])
      
          num_foci = [foc.shape[0] for foc in foci]

  kernels = np.empty((num_studies,31,31,31))
  for i, sample_size in enumerate(sample_sizes):
      temp_uncertainty = 5.7/(2*np.sqrt(2/np.pi)) * np.sqrt(8*np.log(2))
      subj_uncertainty = (11.6/(2*np.sqrt(2/np.pi)) * np.sqrt(8*np.log(2))) / np.sqrt(sample_size)
      smoothing = np.sqrt(temp_uncertainty**2 + subj_uncertainty**2)
      kernels[i,:,:] = kernel_calc(affine, smoothing, 31)

  mb = 1
  for i in range(num_studies):
      mb = mb*(1-np.max(kernels[i]))

  # define bins for histogram
  bin_steps=0.0001
  bin_edges = np.arange(0.00005,1-mb+0.001,bin_steps)
  bin_centers = np.arange(0,1-mb+0.001,bin_steps)
  step = int(1/bin_steps)

  main_start = timer()
  ma = compute_ma(foci, kernels)
  hx = compute_hx(ma, bin_edges)
  ale = compute_ale(ma)
  hx_conv = compute_hx_conv(hx, bin_centers, step)
  z = compute_z(ale, hx_conv, step)
  main_end = timer()
  tfce = compute_tfce(z)
  tfce_end = timer()

  main_time = main_end - main_start
  tfce_time = tfce_end - main_end
  
  return main_time, tfce_time

if __name__ == "__main__":
  rank = MPI.COMM_WORLD.Get_rank()
  size = MPI.COMM_WORLD.Get_size()
  
  num_studies = int(sys.argv[1])
  num_true = int(sys.argv[2])
  rep = int(sys.argv[3])
  
  main_time, tfce_time = ale(num_studies, num_true, rank+(128*rep))

  print(main_time, tfce_time)
  
      
