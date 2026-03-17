
import pyfftw #import before numpy!
import imageio, os

# commom modules
import matplotlib.pyplot as plt
import time
import pickle
from pathlib import Path


from OOPAO.Atmosphere import Atmosphere
from OOPAO.calibration.CalibrationVault import CalibrationVault
from OOPAO.DeformableMirror import DeformableMirror
from OOPAO.MisRegistration  import MisRegistration
from OOPAO.Telescope        import Telescope
from OOPAO.Source           import Source
# calibration modules
from OOPAO.calibration.compute_KL_modal_basis import compute_M2C, compute_KL_basis, compute_M2C_alt

from trwfs.tools.PWFS_CMOS_NOISE_PROOF_may2022_tools import *
from tqdm import trange
from OOPAO.TR_Pyramid import TR_Pyramid
from pypet import Environment, cartesian_product
from trwfs.parameter_files.parameterFile_CMOS_PWFS_aug2022_3 import initializeParameterFile
from OOPAO.tools.displayTools           import displayMap

param = initializeParameterFile()

# %% -----------------------     TELESCOPE   ----------------------------------

# create the Telescope object
tel = Telescope(resolution=param['resolution'], \
                diameter=param['diameter'], \
                samplingTime=param['samplingTime'], \
                centralObstruction=param['centralObstruction'])

# %% -----------------------     NGS   ----------------------------------
# create the Source object
ngs = Source(optBand=param['opticalBand'], \
             magnitude=1.0)

# combine the NGS to the telescope using '*' operator:
ngs * tel

tel.computePSF(zeroPaddingFactor=6)

# %% -----------------------     ATMOSPHERE   ----------------------------------

# create the Atmosphere object
# atm = Atmosphere(telescope=tel, \
#                  r0=param['r0'], \
#                  L0=param['L0'], \
#                  windSpeed=param['windSpeed'], \
#                  fractionalR0=param['fractionnalR0'], \
#                  windDirection=param['windDirection'], \
#                  altitude=param['altitude'])

atm = dummy_atm(tel=tel)
# initialize atmosphere
# atm.initializeAtmosphere(self.tel)
#
# self.atm.update()
#
# self.tel + self.atm
# self.tel.computePSF(8)

# %% -----------------------     DEFORMABLE MIRROR   ----------------------------------
# mis-registrations object
misReg = MisRegistration(param)
# if no coordonates specified, create a cartesian dm

dm = DeformableMirror(telescope=tel, \
                      nSubap=param['nSubaperture'], \
                      mechCoupling=param['mechanicalCoupling'], \
                      misReg=misReg)


# %% -----------------------     Modal Basis   ----------------------------------
# compute the modal basis
# foldername_M2C  = None  # name of the folder to save the M2C matrix, if None a default name is used
# filename_M2C    = None  # name of the filename, if None a default name is used
# # KL Modal basis
# self.M2C_KL = compute_M2C(telescope=self.tel, \
#                      atmosphere=self.atm, \
#                      deformableMirror=self.dm, \
#                      param=self.param, \
#                      nameFolder=None, \
#                      nameFile=None, \
#                      remove_piston=True, \
#                      HHtName=None, \
#                      baseName=None, \
#                      mem_available=8.1e9, \
#                      minimF=False, \
#                      nmo=600, \
#                      ortho_spm=True, \
#                      SZ=np.int64(2 * self.tel.OPD.shape[0]), \
#                      nZer=3, \
#                      NDIVL=1)

M2C_KL = compute_KL_basis(tel=tel, atm=atm, dm=dm)

# M2C_KL = compute_M2C_alt(telescope=tel, \
#                      atmosphere=atm, \
#                      deformableMirror=dm, \
#                      param=None, \
#                      nameFolder=None, \
#                      nameFile=None, \
#                      remove_piston=True, \
#                      HHtName=None, \
#                      baseName=None, \
#                      mem_available=None, \
#                      minimF=False, \
#                      nmo=None, \
#                      ortho_spm=True, \
#                      SZ=np.int(2 * tel.OPD.shape[0]), \
#                      nZer=3, \
#                      NDIVL=1, \
#                      recompute_cov=True, \
#                      save_output=False)

# self.M2C_KL = pickle.load(open("old_code_M2C_KL.pickle", "rb"))


# show the first 10 KL
# dm.coefs = M2C_KL[:, :10]
# tel * dm
# displayMap(tel.OPD)
# plt.show(block=True)



# %% -----------------------     PYRAMID WFS   ----------------------------------

# make sure tel and atm are separated to initialize the PWFS
tel - atm

wfs = TR_Pyramid(nSubap=param['nSubaperture'], \
                  telescope=tel, \
                  modulation=param['modulation'], \
                  lightRatio=param['lightThreshold'], \
                  n_pix_separation=4,
                  calibModulation=param['calibrationModulation'], \
                  psfCentering=param['psfCentering'], \
                  n_pix_edge=2,
                  extraModulationFactor=param['extraModulationFactor'], \
                  postProcessing=param['postProcessing'],
                  nTheta_user_defined = param['nTheta_user_defined'],
				  delta_theta=0)
# %%
tel * wfs


# plt.figure()
# plt.imshow(self.wfs.cam.frame)
# plt.colorbar()
# plt.show(block=True)

#refCube = self.wfs.cam.cube

# %% -----------------------     Time-resolved frames   ----------------------------------
numRemFramesPerBase = np.zeros((M2C_KL.shape[1]))
numRemFramesPerBase[0:100] = [11, 11, 9, 9, 9, 7, 7, 7, 7, 7, 7, 7, 7, 5, 5, 5, 5, 5, 5, 5, 5, 5,
                                      3, 3, 3, 3, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                      0, 0, 0, 0, 0, 0, 0, 0, 0]

# numRemFramesPerBase[0:2] = [9, 9]



remFrames = [0, 1, 3, 5, 7, 9, 11] #np.unique(numRemFramesPerBase)

# %% -----------------------     INTERACTION MATRIX   ----------------------------------


# Modal Interaction Matrix
from OOPAO.calibration.InteractionMatrix import InteractionMatrix, interactionMatrix_withTR, interactionMatrix_withTR_no_negative, interactionMatrix_withTR_withintmat

foldername_M2C = None  # name of the folder to save the M2C matrix, if None a default name is used
filename_M2C = None  # name of the filename, if None a default name is used

# M2C = compute_M2C(param=self.param, \
#                   telescope=self.tel, \
#                   atmosphere=self.atm, \
#                   deformableMirror=self.dm, \
#                   nameFolder=None, \
#                   nameFile=None, \
#                   remove_piston=True)

# amplitude of the modes in m
stroke=1*1e-9
# Modal Interaction Matrix
M2C = M2C_KL[:,:param['nModes']]



# project the mode on the DM
# dm.coefs = M2C_KL[:, 1] * 1*(1e-9)
a = np.zeros(M2C_KL.shape[1])
a[1] = 500*(1e-9)
dm.coefs = np.matmul(M2C_KL, a)

tel * dm
displayMap(tel.OPD)
dm_tel_OPD = tel.OPD
rms = np.sqrt(np.sum((tel.OPD**2) )/ np.sum(tel.pupil))
print(f"DM: max={np.max(tel.OPD)}, min={np.min(tel.OPD)}, rms={rms}")

dm.coefs = 0
tel * dm

bases = generateBases(25, tel.resolution, "KL", display=False)
phaseMask = bases[0,:,:]*500*(1e-9)
atm = dummy_atm(tel)
atm.OPD_no_pupil = phaseMask
atm.OPD = atm.OPD_no_pupil*tel.pupil
tel+atm
#
rms = np.sqrt(np.sum((tel.OPD**2) )/ np.sum(tel.pupil))
print(f"ATM: max={np.max(tel.OPD)}, min={np.min(tel.OPD)}, rms={rms}")
fig = plt.figure(10)
atm_tel_OPD = tel.OPD
phasemask_view = plt.imshow(tel.OPD)
plt.colorbar(phasemask_view)
tel-atm

fig = plt.figure(11)
diff_view = plt.imshow(atm_tel_OPD-dm_tel_OPD)
plt.colorbar(diff_view)
plt.show(block=True)


bases = generateBases(M2C.shape[1], tel.resolution, "KL", display=False)
custom_int_mat_D_plus, custom_int_mat_D = generateCustomInteractionMatrixFromDM(dm, tel, wfs, M2C_KL, numRemFramesPerBase, D_amp=stroke)


# custom_atm_mat_D_plus = generateCustomInteractionMatrix(atm, tel, wfs, bases, numRemFramesPerBase, D_amp=stroke)
custom_atm_mat_D_plus = custom_int_mat_D_plus
# normal_atm_mat_D = generateInteractionMatrix(atm, tel, wfs, bases, removed_frames=None, amplitude=stroke)
# normal_atm_mat_D_plus = inv(normal_atm_mat_D.T @ normal_atm_mat_D) @ normal_atm_mat_D.T
normal_atm_mat_D_plus =  custom_atm_mat_D_plus

# the major difference here is that instead of using the atm, im using the DM. 
ao_calib_custom, intmat_custom = interactionMatrix_withTR_withintmat(ngs=ngs,
                                    atm=atm,
                                    tel=tel,
                                    dm=dm,
                                    wfs=wfs,
                                    M2C=M2C_KL,
                                    stroke=stroke,
                                    phaseOffset=0,
                                    nMeasurements=1,
                                    noise='off',
                                    custom_frames=True,
                                    custom_remove_frames=numRemFramesPerBase[0:100])



# M2C_zonal = np.eye(dm.nValidAct)
# # zonal interaction matrix
# calib_zonal = interactionMatrix_withTR(  ngs            = ngs,\
#                             atm            = atm,\
#                             tel            = tel,\
#                             dm             = dm,\
#                             wfs            = wfs,\
#                             M2C            = M2C_zonal,\
#                             stroke         = stroke,\
#                             nMeasurements  = 1,\
#                             noise          = 'off',
#                             custom_frames=True,
#                             custom_remove_frames=numRemFramesPerBase[0:100])
#
# calib_KL = CalibrationVault(calib_zonal.D@M2C_KL)

ao_calib_normal = InteractionMatrix(ngs=ngs,
                                    atm=atm,
                                    tel=tel,
                                    dm=dm,
                                    wfs=wfs,
                                    M2C=M2C_KL,
                                    stroke=stroke,
                                    phaseOffset=0,
                                    nMeasurements=1,
                                    noise='off')





#
# show the modes projected on the dm, cropped by the pupil and normalized by their maximum value
# displayMap(self.tel.OPD, norma=True)
# plt.title('Basis projected on the DM')



# These are the calibration data used to close the loop

M2C_CL = M2C_KL

# combine telescope with atmosphere
# tel + atm

# initialize DM commands
dm.coefs = 0
ngs * tel * dm * wfs


# plt.figure(1)
# plt.imshow(wfs.cam.frame)
#
# plt.figure(2)
# plt.imshow(np.sum(wfs.cam.cube, axis=0))
# plt.show(block=True)

custom_D = intmat_custom
custom_D_plus = inv(custom_D.T @ custom_D) @ custom_D.T
normal_D = ao_calib_normal.D
normal_D_plus = inv(normal_D.T @ normal_D) @ normal_D.T

iterations = 1000

custom_error = np.zeros((iterations,M2C_CL.shape[1]))
normal_error = np.zeros((iterations,M2C_CL.shape[1]))
full_custom_error = np.zeros((iterations,M2C_CL.shape[1]))
atm_custom_error = np.zeros((iterations,M2C_CL.shape[1]))
atm_normal_error = np.zeros((iterations,M2C_CL.shape[1]))
zonal_error = np.zeros((iterations,M2C_CL.shape[1]))

err_prop_normal = np.zeros(iterations)
err_prop_custom = np.zeros(iterations)





# loop parameters
gainCL = param['gainCL']
wfs.cam.photonNoise = False
tel - atm

# a = generateA(M2C_CL.shape[1], totalDistortion=200 * (1e-9))
a = np.zeros(M2C_CL.shape[1])
a[1] = 5 * (1e-9)
dm.coefs = np.matmul(M2C_CL, a)

for i in range(iterations):



    """The distortion associated to the 'a' vector"""
    # Phi_distortion = np.zeros(bases.shape)
    # for j in range(len(a)):
    #     Phi_distortion[j, :, :] = a[j] * bases[j, :, :]
    #
    # Phi_distortion = np.sum(Phi_distortion, axis=0)
    #
    # atm.OPD_no_pupil = Phi_distortion
    # atm.OPD = atm.OPD_no_pupil * tel.pupil
    #
    # tel + atm

    a_est_custom = np.zeros((M2C_CL.shape[1]))
    a_est_normal = np.zeros((M2C_CL.shape[1]))
    a_est_full_custom = np.zeros((M2C_CL.shape[1]))
    a_est_atm_custom = np.zeros((M2C_CL.shape[1]))
    a_est_atm_normal = np.zeros((M2C_CL.shape[1]))
    a_est_zonal = np.zeros((M2C_CL.shape[1]))

    dm.coefs = np.matmul(M2C_CL, a)

    ngs * tel * dm * wfs
    # print(f"delta_theta={wfs.delta_theta}, delta_tip={wfs.delta_Tip}, delta_tilt={wfs.delta_Tilt}")
    for j_mode in range(len(a_est_custom)):
        d_idx = np.argwhere(remFrames == numRemFramesPerBase[j_mode])[0][0]
        a_est_custom[j_mode] = (custom_D_plus @ wfs.signalCube[d_idx,:])[j_mode]


    for j_mode in range(len(a_est_full_custom)):
        d_idx = np.argwhere(remFrames == numRemFramesPerBase[j_mode])[0][0]
        # signal = wfs.pyramidSignalCube_2D[d_idx,:,:]
        # signal_f = signal.flatten()
        signal_f = wfs.signalCube[d_idx, :]
        a_est_full_custom[j_mode] = (custom_int_mat_D_plus @ signal_f)[j_mode]

    for j_mode in range(len(a_est_full_custom)):
        d_idx = np.argwhere(remFrames == numRemFramesPerBase[j_mode])[0][0]
        # signal = wfs.pyramidSignalCube_2D[d_idx,:,:]
        # signal_f = signal.flatten()
        signal_f = wfs.signalCube[d_idx, :]
        a_est_atm_custom[j_mode] = (custom_atm_mat_D_plus @ signal_f)[j_mode]

    # for j_mode in range(len(a_est_zonal)):
    #     d_idx = np.argwhere(remFrames == numRemFramesPerBase[j_mode])[0][0]
    #     a_est_zonal[j_mode] = (M2C_CL @ (calib_zonal.M @ wfs.signalCube[d_idx,:])[j_mode])

    a_est_normal = normal_D_plus @ wfs.signalCube[0, :]
    a_est_atm_normal = normal_atm_mat_D_plus @ wfs.signalCube[0, :]
    cur_coef = dm.coefs.copy()
    dm.coefs = cur_coef - np.matmul(M2C_CL, a_est_normal)
    err_prop_normal[i] = a[1] / a_est_normal[1]
    err_prop_custom[i] = a[1] / a_est_custom[1]

    # tel * dm
    # displayMap(tel.OPD)
    # plt.show(block=True)

    dm.coefs = cur_coef - np.matmul(M2C_CL, a_est_full_custom)

    # tel * dm
    # displayMap(tel.OPD)
    # plt.show(block=True)


    custom_error[i, :] = np.abs(a-a_est_custom)
    normal_error[i, :] = (np.abs(a-a_est_normal))
    full_custom_error[i, :] = (np.abs(a-a_est_full_custom))
    atm_custom_error[i, :] = (np.abs(a - a_est_atm_custom))
    atm_normal_error[i, :] = (np.abs(a - a_est_atm_normal))
    zonal_error[i, :] = (np.abs(a - a_est_zonal))

print(f"err_prop={np.mean(err_prop_normal)}, err_prop_custom={np.mean(err_prop_custom)}")

# print(f"custom error = {np.mean(custom_error)}, "
#       f"normal_error={np.mean(normal_error)}, "
#       f"full_custom_error={np.mean(full_custom_error)}, "
#       f"atm_custom_error={np.mean(atm_custom_error)},"
#       f"atm_normal_error={np.mean(atm_normal_error)},"
#       f"zonal_error={np.mean(zonal_error)}")


plt.figure(1)
plt.plot(np.mean(normal_error, axis=0), label='normal')
plt.plot(np.mean(custom_error, axis=0), label="custom_1")
plt.plot(np.mean(full_custom_error, axis=0), label="custom_2")

plt.legend()

plt.show(block=True)