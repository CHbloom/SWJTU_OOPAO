from trwfs.sim.PWFS_CL_batch_pypet import PWFS_CL
from trwfs.tools.PWFS_CMOS_NOISE_PROOF_may2022_tools import dict_to_trajectory
from pypet import Environment, cartesian_product

from trwfs.parameter_files.parameterFile_TR_PWFS_general import *

folderDataSaveLocation = "/home/simonc/Documents/Programming/OOPAO_TRWFS/trwfs/sum/res/"


"""This file serves to generate all simulation results"""

"""Results about noise propagation"""

"""Sensitivity as a function of the mode"""

"""Delta I per modulation frame"""

"""Modulation animation"""



"""Reconstruction of fixed parameters"""

"""Reconstruction of random parameters"""



"""Residual error per loop"""

"""Residual error per magnitude of source"""

"""Residual error vs photons per subaperture"""

"""Strehl ratio vs photon per subaperture"""

PC = PWFS_CL(minimum_data_only=True)

param = initializeParameterFile()

# Create an environment that handles running our simulation
env = Environment(trajectory='run_loops', filename=folderDataSaveLocation + 'quick_test.hdf5',
                  file_title='PWFS_CL_ppsub_strehl',
                  comment='Simulation with an r0 of 0.186, subaperture photons and modulation=0 and 5.',
                  large_overview_tables=True,
                  log_config='DEFAULT',
                  log_stdout=True,
                  overwrite_file=True)

# Get the trajectory from the environment
traj = env.trajectory

dict_to_trajectory(param, traj)


# Add  parameters
traj.f_add_parameter('enable_custom_frames', True, comment='enable_custom_frames')
traj.f_add_parameter('ao_calib_file', "", comment='ao_calib_file')

# Explore the parameters with a cartesian product

traj.f_explore(cartesian_product({'magnitude': [10.0, 12.0, 15.0],
                                  'gainCL': [0.1,  0.2, 0.3],
                                  'nTheta_user_defined': [48, 48, 48],
                                  'modulation': [0, 5, 5],
                                  'enable_custom_frames': [False, False, True],
                                  'ao_calib_file': ["ao_calib_file_nomod_test_509.pickle","ao_calib_file_normal_mod5_48F_mag0_test_509pickle", "ao_calib_file_custom_mod5_48F_mag0_test_509.pickle"]},
                                 ('magnitude', 'gainCL', ('nTheta_user_defined', 'modulation', 'enable_custom_frames', 'ao_calib_file'))))

#traj.f_explore(cartesian_product({'magnitude': [8.0, 9.0],
#                                  'gainCL': [0.4],
#                                  'nTheta_user_defined': [ 48],
#                                  'modulation': [0],
#                                  'enable_custom_frames': [False],
#                                  'ao_calib_file': ["ao_calib_file_normal_mod0_48F_mag0_test_506.pickle"]},
#                                 ('magnitude', 'gainCL', ('nTheta_user_defined', 'modulation', 'enable_custom_frames', 'ao_calib_file'))))




# Run the simulation with all parameter combinations
env.run(PC.run_closed_loop)

# Let's check that all runs are completed!
assert traj.f_is_completed()

# Finally disable logging and close all log-files
env.disable_logging()

