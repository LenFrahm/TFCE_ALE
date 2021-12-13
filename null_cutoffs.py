import numpy as np
from utils.compute import compute_null_cutoffs
from utils.template import sample_space, shape
import sys
import h5py

if __name__ == "__main__":
    num_studies = int(sys.argv[1])
    num_true = int(sys.argv[2])
    rep = int(sys.argv[3])
    
    with h5py.File(f"/p/scratch/cinm-7/pyALE_TFCE/output/main_{num_studies}_offset.gzip", "r") as main:
        hx_conv = main[f"hx_conv/{num_true}/{rep}"][:]
        kernels = main[f"kernels/{num_true}/{rep}"][:]
        num_foci = main[f"num_foci/{num_true}/{rep}"][:]

    for i in range(5000):   
        null_max_ale, null_max_cluster, null_max_tfce = compute_null_cutoffs(sample_space,
                                                                                       num_foci,
                                                                                       kernels,
                                                                                       hx_conv=hx_conv,
                                                                                       thresh=0.001)
        print(null_max_ale, null_max_cluster, null_max_tfce)
        