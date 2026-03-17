
import pyfftw #import before numpy!
import imageio, os

# commom modules
import matplotlib.pyplot as plt
import time
import pickle
from pathlib import Path


from OOPAO.Atmosphere import Atmosphere

from OOPAO.DeformableMirror import DeformableMirror
from OOPAO.MisRegistration  import MisRegistration
from OOPAO.Telescope        import Telescope
from OOPAO.Source           import Source
# calibration modules
from OOPAO.calibration.compute_KL_modal_basis import compute_M2C, compute_KL_basis

from trwfs.tools.PWFS_CMOS_NOISE_PROOF_may2022_tools import *
from tqdm import trange
from OOPAO.TR_Pyramid import TR_Pyramid
from pypet import Environment, cartesian_product
from trwfs.parameter_files.parameterFile_CMOS_PWFS_aug2022_3 import initializeParameterFile

class PWFS_CL():

    def __init__(self, minimum_data_only=False):
        self.DATA = {}
        self.minimum_data_only = minimum_data_only


    def setup(self, traj, enable_custom_frames, ao_calib_file=None):

        self.param = traj.parameters.f_to_dict(fast_access=True, short_names=True)
        # %% -----------------------     TELESCOPE   ----------------------------------

        # create the Telescope object
        self.tel = Telescope(resolution=self.param['resolution'], \
                        diameter=self.param['diameter'], \
                        samplingTime=self.param['samplingTime'], \
                        centralObstruction=self.param['centralObstruction'])

        # %% -----------------------     NGS   ----------------------------------
        # create the Source object
        self.ngs = Source(optBand=self.param['opticalBand'], \
                     magnitude=self.param['magnitude'])

        # combine the NGS to the telescope using '*' operator:
        self.ngs * self.tel

        self.tel.computePSF(zeroPaddingFactor=6)

        # %% -----------------------     ATMOSPHERE   ----------------------------------

        # create the Atmosphere object
        self.atm = Atmosphere(telescope=self.tel, \
                         r0=self.param['r0'], \
                         L0=self.param['L0'], \
                         windSpeed=self.param['windSpeed'], \
                         fractionalR0=self.param['fractionnalR0'], \
                         windDirection=self.param['windDirection'], \
                         altitude=self.param['altitude'])
        # initialize atmosphere
        self.atm.initializeAtmosphere(self.tel)

        self.atm.update()

        self.tel + self.atm
        self.tel.computePSF(8)

        # %% -----------------------     DEFORMABLE MIRROR   ----------------------------------
        # mis-registrations object
        misReg = MisRegistration(self.param)
        # if no coordonates specified, create a cartesian dm
        self.dm = DeformableMirror(telescope=self.tel, \
                              nSubap=self.param['nSubaperture'], \
                              mechCoupling=self.param['mechanicalCoupling'], \
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

        self.M2C_KL = compute_KL_basis(tel=self.tel, atm=self.atm, dm=self.dm)

        # self.M2C_KL = pickle.load(open("old_code_M2C_KL.pickle", "rb"))


        # show the first 10 KL
        self.dm.coefs = self.M2C_KL[:, :10]
        self.tel * self.dm
        # displayMap(self.tel.OPD)

        if not self.minimum_data_only:
            traj.f_add_result('dm_OPD', self.dm.OPD, comment="OPD of the DM, with the first 10 modes")
            traj.f_add_result('tel_OPD', self.tel.OPD, comment="OPD of the telescope, with the first 10 modes")



        # bases = generateBases(300, self.tel.resolution, 'KL', display=False, scale=False)
        # bases_flat = np.reshape(bases, (300, self.tel.resolution**2)).T
        # #self.dm.coefs = bases_flat[:, :10]
        # self.tel * self.dm
        # displayMap(self.tel.OPD)

        # plt.show(block=True)

        # %% -----------------------     PYRAMID WFS   ----------------------------------

        # make sure tel and atm are separated to initialize the PWFS
        self.tel - self.atm

        self.wfs = TR_Pyramid(nSubap=self.param['nSubaperture'], \
                              telescope=self.tel, \
                              modulation=self.param['modulation'], \
                              lightRatio=self.param['lightThreshold'], \
                              n_pix_separation=4,
                              calibModulation=self.param['calibrationModulation'], \
                              psfCentering=self.param['psfCentering'], \
                              n_pix_edge=1,
                              extraModulationFactor=self.param['extraModulationFactor'], \
                              postProcessing=self.param['postProcessing'],
                              nTheta_user_defined = self.param['nTheta_user_defined'])
        # %%
        self.tel * self.wfs

        subArea = (self.wfs.telescope.D / self.wfs.nSubap) ** 2 # taille d'une sous-ouverture en m2
        traj.f_add_result('subArea', subArea, comment="Size of a sub aperture in m2")
        photons_per_subArea = self.wfs.telescope.src.nPhoton * self.wfs.telescope.samplingTime * subArea
        traj.f_add_result('photons_per_subArea', photons_per_subArea, comment="Photons per sub aperture")

        # plt.figure()
        # plt.imshow(self.wfs.cam.frame)
        # plt.colorbar()
        # plt.show(block=True)

        #refCube = self.wfs.cam.cube

        # %% -----------------------     Time-resolved frames   ----------------------------------
        self.numRemFramesPerBase = np.zeros((306))
        if traj.enable_custom_frames == True:
            print("Finding optimal Frames to remove per mode")
            with silence():
                # self.dm.coefs = self.M2C_KL[:, :100]
                # self.tel * self.dm
                # bases = self.tel.OPD
                # bases = np.transpose(bases, axes=[2, 0, 1])
                # self.numRemFramesPerBase[0:100] = findOptimalFrames(self.tel, self.wfs,
                #                                                     nTheta_user_defined=self.param["nTheta_user_defined"],
                #                                                     bases=bases)
                # self.numRemFramesPerBase[0:100] = findOptimalFramesRemovedPerMode(self.tel,
                #                                                               self.wfs,
                #                                                               nTheta_user_defined=self.param["nTheta_user_defined"],
                #                                                               numBases=100)
                self.numRemFramesPerBase[0:100] = [11, 11, 9, 9, 9, 7, 7, 7, 7, 7, 7, 7, 7, 5, 5, 5, 5, 5, 5, 5, 5, 5,
                                                  3, 3, 3, 3, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                                                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                  0, 0, 0, 0, 0, 0, 0, 0, 0]
        traj.f_add_result("numRemFramesPerBase", self.numRemFramesPerBase,
                          comment="Number of removed frames per base (first 100 modes)")
        self.remFrames = np.unique(self.numRemFramesPerBase)

        # %% -----------------------     INTERACTION MATRIX   ----------------------------------

        # amplitude of the modes in m
        stroke = 1e-9
        # Modal Interaction Matrix
        from OOPAO.calibration.InteractionMatrix import InteractionMatrix, interactionMatrix_withTR
        from OOPAO.calibration.ao_calibration import CalibrationVault

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
        stroke=1e-9
        # Modal Interaction Matrix
        M2C = self.M2C_KL[:,:self.param['nModes']]

        if ((ao_calib_file) and (os.path.exists(ao_calib_file))):
            ao_calib = pickle.load(open(ao_calib_file, "rb"))
            print("Used provided interaction matrix")

        else:

            M2C_zonal = np.eye(self.dm.nValidAct)
            # zonal interaction matrix
            calib_zonal = interactionMatrix_withTR(ngs=self.ngs, \
                                            atm=self.atm, \
                                            tel=self.tel, \
                                            dm=self.dm, \
                                            wfs=self.wfs, \
                                            M2C=M2C_zonal, \
                                            stroke=stroke, \
                                            nMeasurements=1, \
                                            noise='off',
                                           custom_frames=enable_custom_frames,
                                           custom_remove_frames=self.numRemFramesPerBase[0:100])
            print("No interaction matrix provided, will calculate one")

            if ao_calib_file:
                pickle.dump(ao_calib, open(ao_calib_file, "wb"))

        traj.f_add_result('OpticalGain', np.std(ao_calib.D, axis=0), comment="Optical Gain as a function of the mode number of the interaction matrix")

        D_plus = inv(ao_calib.D.T @ ao_calib.D) @ ao_calib.D.T
        B = D_plus @ D_plus.T
        traj.f_add_result('NoisePropagation', np.diag(B), comment="Noise propagation as a function of the mode number of the interaction matrix")

        calib_kl = CalibrationVault(calib_zonal.D @ self.M2C_KL)

        # project the mode on the DM
        self.dm.coefs = self.M2C_KL[:, :100]

        self.tel * self.dm
        #
        # show the modes projected on the dm, cropped by the pupil and normalized by their maximum value
        # displayMap(self.tel.OPD, norma=True)
        # plt.title('Basis projected on the DM')

        KL_dm = np.reshape(self.tel.OPD, [self.tel.resolution ** 2, self.tel.OPD.shape[2]])

        covMat = (KL_dm.T @ KL_dm) / self.tel.resolution ** 2


        # These are the calibration data used to close the loop
        self.calib_CL = ao_calib
        self.M2C_CL = self.M2C_KL

        # combine telescope with atmosphere
        self.tel + self.atm

        # initialize DM commands
        self.dm.coefs = 0
        self.ngs * self.tel * self.dm * self.wfs


    def run_closed_loop(self, traj):

        self.setup(traj, traj.enable_custom_frames, traj.ao_calib_file)


        SR = np.zeros(self.param['nLoop'])
        SR_H = np.zeros(self.param['nLoop'])
        total = np.zeros(self.param['nLoop'])
        residual = np.zeros(self.param['nLoop'])
        OPD = np.zeros((self.param['nLoop'], self.tel.OPD.shape[0], self.tel.OPD.shape[1]))
        wfsSignal_record = np.zeros((self.param['nLoop'], ))
        if traj.enable_custom_frames:
            wfsSignal = np.zeros(self.wfs.pyramidSignalCube.shape)

        else:
            wfsSignal = np.zeros(self.wfs.pyramidSignal.shape)

        wfsSignal_record = np.zeros((self.param['nLoop'],)+wfsSignal.shape)

        a_est = np.zeros((306))
        wfs_to_use = None

        # loop parameters
        gainCL = self.param['gainCL']
        self.wfs.cam.photonNoise = self.param['photonNoise']

        reconstructor = self.M2C_CL @ self.calib_CL.M

        if not self.minimum_data_only:
            traj.f_add_result("InteractionMatrix", self.calib_CL.M, comment="Interaction matrix")
            traj.f_add_result("M2C", self.M2C_CL, comment="Modes 2 Commands (M2C) for the DM")

        for i in trange(self.param['nLoop'], position=0, leave=True):
        #for i in range(self.param['nLoop']):
            #a = time.time()
            # update phase screens => overwrite tel.OPD and consequently tel.src.phase
            self.atm.update()
            # save phase variance
            total[i] = np.std(self.tel.OPD[np.where(self.tel.pupil > 0)]) * 1e9
            # save turbulent phase
            turbPhase = self.tel.src.phase

            # propagate to the WFS with the CL commands applied
            self.tel * self.dm * self.wfs

            # save the DM OPD shape
            dmOPD = self.tel.pupil * self.dm.OPD * 2 * np.pi / self.ngs.wavelength



            if traj.enable_custom_frames:
                for j_mode in range(len(a_est)):
                    d_idx = np.argwhere(self.remFrames == self.numRemFramesPerBase[j_mode])[0][0]
                    a_est[j_mode] = (self.calib_CL.M @ wfsSignal[d_idx, :])[j_mode]
            else:
                a_est = self.calib_CL.M @ wfsSignal

            correction = np.matmul(self.M2C_CL, a_est)

            self.dm.coefs = self.dm.coefs - gainCL * correction
            # store the slopes after computing the commands => 2 frames delay
            if traj.enable_custom_frames:
                wfsSignal = self.wfs.pyramidSignalCube
            else:
                wfsSignal = self.wfs.pyramidSignal


            SR[i] = np.exp(-np.var(self.tel.src.phase[np.where(self.tel.pupil == 1)]))
            phase_wave = self.tel.src.phase[np.where(self.tel.pupil == 1)] * self.tel.src.wavelength / 2 / np.pi
            rms = np.std(phase_wave)
            test = np.exp(-4*(np.pi**2)*(rms**2) / ( (0.790e-6)**2))
            wavelenth = 1.654e-6 #H band
            SR_H[i] = np.exp(-4*(np.pi**2)*(rms**2) / (wavelenth**2))
            residual[i] = np.std(self.tel.OPD[np.where(self.tel.pupil > 0)]) * 1e9
            OPD[i, :, :] = self.tel.OPD
            if traj.enable_custom_frames:
                wfsSignal_record[i,:,:] = wfsSignal
            else:
                wfsSignal_record[i,:] = wfsSignal



        traj.f_add_result('SR', SR, comment="Strehl Ratio")
        traj.f_add_result('SR_H', SR_H, comment="Strehl Ratio in the H band")
        traj.f_add_result('Residual', residual, comment="Residual")
        traj.f_add_result('Turbulence', total, comment="Turbulence")
        if not self.minimum_data_only:
            traj.f_add_result('wfsSignal', wfsSignal_record, comment="wfsSignal during closed loop")
        #traj.f_add_result('OPD', OPD, comment="OPD")


def dict_to_trajectory(d, traj):
    for k,v in d.items():
        traj.f_add_parameter(k, v, comment=k)







if __name__ == "__main__":
    PC = PWFS_CL()

    param = initializeParameterFile()

    # Create an environment that handles running our simulation
    env = Environment(trajectory='run_loops', filename='/home/cars2019/DATA/HDF/testing_noise.hdf5',
                      file_title='PWFS_CL_frames',
                      comment='Trying to see if the interaction matrix is being done correctly',
                      large_overview_tables=True,
                      log_config='DEFAULT',
                      # git_repository="../",
                      # git_message="Auto-commit for run.",
                      log_stdout=True,
                      overwrite_file=True)

    # Get the trajectory from the environment
    traj = env.trajectory

    dict_to_trajectory(param, traj)


    # Add  parameters
    #traj.f_add_parameter('param', param, comment='param')
    traj.f_add_parameter('enable_custom_frames', True, comment='enable_custom_frames')
    # traj.f_add_parameter('custom_frame_type', "binary", comment='enable_custom_frames')
    traj.f_add_parameter('ao_calib_file', "", comment='ao_calib_file')

    # Explore the parameters with a cartesian product

    traj.f_explore(cartesian_product({'magnitude': [12.0, 12.5, 13.0, 13.5, 14.0, 14.5, 15.0],
                                      'gainCL': [0.4],
                                      'nTheta_user_defined': [48],
                                      'modulation': [5],
                                      'enable_custom_frames': [True]},
                                      #'ao_calib_file': ["ao_calib_file_custom_48F_mag0_s30.pickle", "ao_calib_file_normal_48F_mag0_s30.pickle"]},
                                     ('magnitude', 'gainCL', ('nTheta_user_defined', 'modulation', 'enable_custom_frames'))))

    # traj.f_explore(cartesian_product({'magnitude': [10, 11,12, 13, 14, 15],
    #                                   'gainCL': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    #                                   'enable_custom_frames': [True, False],
    #                                   'ao_calib_file': ["ao_calib_file_custom_48F_mag0_s13.pickle", "ao_calib_file_normal_48F_mag0_s13.pickle"]},
    #                                  ('magnitude', 'gainCL', ('enable_custom_frames', 'ao_calib_file'))))


    # traj.f_explore(cartesian_product({'magnitude': [14, 15],
    #                                   'gainCL': [0.1, 0.2],
    #                                   'enable_custom_frames': [True, False]}))

    # Run the simulation with all parameter combinations
    env.run(PC.run_closed_loop)

    # Let's check that all runs are completed!
    assert traj.f_is_completed()

    # Finally disable logging and close all log-files
    env.disable_logging()

    # PC.run_loops(param=param,
    #              mags=mags,
    #             loop_gains=loop_gains,
    #             enable_custom_frames=True,
    #             iterations=1000,
    #             filename="mag_and_gain_vs_residual_data_custom_fixed_0L50_testing4.pickle",
    #             ao_calib_file=None)
    # #
    # param['modulation'] = 0
    # PC.run_loops(param=param,
    #              mags=mags,
    #              loop_gains=loop_gains,
    #              enable_custom_frames=False,
    #              iterations=1000,
    #              filename="mag_and_gain_vs_residual_data_normal_fixed_0L50_testing4.pickle",
    #              ao_calib_file=None)
