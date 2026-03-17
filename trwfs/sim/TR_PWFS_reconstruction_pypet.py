from trwfs.tools.PWFS_CMOS_NOISE_PROOF_may2022_tools import *
import pickle

from OOPAO.TRM_Pyramid          import TRM_Pyramid
from pypet import Environment, cartesian_product


class TRPWFS():

    def __init__(self, param):
        self.param = param
        # %% -----------------------     TELESCOPE   ----------------------------------
        # create the Telescope object
        self.tel = Telescope(resolution=param['resolution'],
                        diameter=param['diameter'],
                        samplingTime=param['samplingTime'],
                        centralObstruction=param['centralObstruction'])

        # %% -----------------------     NGS   ----------------------------------
        # create the Source object
        self.ngs = Source(optBand=param['opticalBand'], \
                     magnitude=param['magnitude'])

        # combine the NGS to the telescope using '*' operator:
        self.ngs * self.tel

        self.tel.computePSF(zeroPaddingFactor=6)

        # %% -----------------------     ATMOSPHERE   ---------------------------------

        "Dummy atmosphere to be able to fit whatever aberration we want"
        self.atm = dummy_atm(self.tel)

        self.numBases = param["nModes"]
        self.bases = generateBases(num=param["nModes"], res=self.tel.resolution, baseType="KL", display=False, scale=False)


        # %% -----------------------     PYRAMID WFS   ----------------------------------


        self.wfs = TRM_Pyramid(nSubap=param['nSubaperture'],
                      telescope=self.tel,
                      modulation=param['modulation'],
                      lightRatio=param['lightThreshold'],
                      n_pix_separation=4,
                      calibModulation=param['calibrationModulation'],
                      psfCentering=param['psfCentering'],
                      n_pix_edge=4,
                      extraModulationFactor=param['extraModulationFactor'],
                      postProcessing=param['postProcessing'],
                      nTheta_user_defined=param["nTheta_user_defined"],
                      temporal_weights_settings=None)

        self.mode = param.recon_mode






    def simulateReconstructionError(self, D_amp, a_amp, iterations):


        self.a_err_noisy = np.zeros((iterations))
        self.a_est_noisy = np.zeros((iterations, self.numBases))
        self.a_noisy = np.zeros((iterations, self.numBases))




        Dplus = None

        if self.mode == "binary_opt":
            custom_remove_frames = findOptimalFramesRemovedPerMode(self.tel, self.wfs, self.wfs.nTheta, self.numBases)

        weights = np.zeros((self.wfs.nTheta, self.numBases))


        cube_ref = None
        for i in tqdm(range(iterations)):
            force_a = generateA(self.numBases, (50*1e-9))
            cube = None
            cube_noisy = None
            if self.mode == "normal":
                self.a_noisy[i, :], self.a_est_noisy[i, :], self.a_err_noisy[i], Dplus, cube_noisy, cube_ref = calcReconstructionError(self.atm, self.tel, self.wfs,
                                                                                                    self.bases,
                                                                                                    removed_frames=None,
                                                                                                    D_amp=D_amp,
                                                                                                    a_amp=a_amp,
                                                                                                    display=False,
                                                                                                    noise=True,
                                                                                                    seed=i * 42,
                                                                                                    force_a=force_a,
                                                                                                    force_Dplus=Dplus,
                                                                                                    forceCube=cube_noisy,
                                                                                                    forceRefCube=cube_ref)

            if self.mode == "binary_opt":

                self.a_noisy[i, :], self.a_est_noisy[i, :], self.a_err_noisy[i], Dplus = calcReconstructionErrorMultiD(self.atm, self.tel, self.wfs,
                                                                     self.bases,
                                                                     custom_remove_frames,
                                                                     D_amp=D_amp,
                                                                     a_amp=a_amp,
                                                                     display=False,
                                                                     noise=True,
                                                                     seed=i * 42,
                                                                     force_a=force_a,
                                                                     force_Dplus=Dplus,
                                                                     forceCube=cube_noisy,
                                                                     forceRefCube=cube_ref)

            elif self.mode == "weighted":
                self.a_noisy[i, :], self.a_est_noisy[i, :], self.a_err_noisy[i], Dplus, weights = calcReconstructionErrorWithWeights(self.atm, self.tel, self.wfs,
                                                                                          self.bases,
                                                                                          weights,
                                                                                          D_amp=D_amp,
                                                                                          a_amp=a_amp,
                                                                                          display=False,
                                                                                          noise=True,
                                                                                          seed=i * 42,
                                                                                          force_a=force_a,
                                                                                          force_Dplus=Dplus,
                                                                                          forceCube=cube_noisy,
                                                                                          forceRefCube=cube_ref)

            else:
                pass





    def getErrorProgapation(self, numFramesRemoved, D_amp):

        B_trs = np.zeros((len(numFramesRemoved), self.numBases))

        for i in range(len(numFramesRemoved)):
            removed_frames = calcEquidistantFrameIndices(numFramesRemoved[i], self.wfs.nTheta)
            B_trs[i,:] = calcBDiag(self.atm, self.tel, self.wfs, self.bases, removed_frames=removed_frames, D_amp=D_amp)

        return B_trs

    def calcInteractionMatrix(self, removed_frames=None, D_amp=(1e-9)):
        return generateInteractionMatrix(self.atm, self.tel, self.wfs, self.bases, removed_frames=removed_frames, amplitude=D_amp)

    def getDeltaIPerFrame(self, D_amp):
        deltaI = np.zeros((self.param["nTheta_user_defined"], self.numBases))
        modulation_angle = np.zeros((self.param["nTheta_user_defined"], self.numBases))

        for i in range(self.numBases):
            # Start by making the flat (no distortion image)
            self.tel - self.atm
            self.tel * self.wfs
            cube_ref_no_remove = self.wfs.cam.cube

            # Get phase mask on atmosphere
            self.atm.OPD_no_pupil = self.bases[i, :, :] * D_amp
            self.atm.OPD = self.atm.OPD_no_pupil * self.tel.pupil
            # Apply atmosphere to telescope
            self.tel + self.atm
            # Propagate to wfs
            self.tel * self.wfs
            # Grab image
            cube_no_remove = self.wfs.cam.cube
            # Clear atmosphere
            self.tel - self.atm
            deltaI[:, i] = calcDeltaIPerFrame(cube_no_remove, cube_ref_no_remove)
            modulation_angle[:, i] = self.wfs.thetaModulation

        return deltaI, modulation_angle


    def getDeltaIPerPixel(self, D_amp):
        deltaI = np.zeros((self.wfs.cam.cube.shape[0], self.wfs.cam.cube.shape[1], self.param["nTheta_user_defined"], self.numBases))
        modulation_angle = np.zeros((self.param["nTheta_user_defined"], self.numBases))

        for i in range(self.numBases):
            # Start by making the flat (no distortion image)
            self.tel - self.atm
            self.tel * self.wfs
            cube_ref_no_remove = self.wfs.cam.cube

            # Get phase mask on atmosphere
            self.atm.OPD_no_pupil = self.bases[i, :, :] * D_amp
            self.atm.OPD = self.atm.OPD_no_pupil * self.tel.pupil
            # Apply atmosphere to telescope
            self.tel + self.atm
            # Propagate to wfs
            self.tel * self.wfs
            # Grab image
            cube_no_remove = self.wfs.cam.cube
            # Clear atmosphere
            self.tel - self.atm
            deltaI[:, :, :, i] = np.abs((cube_no_remove/np.sum(cube_no_remove, axis=0)[:, np.newaxis]) -  (cube_ref_no_remove/np.sum(cube_ref_no_remove, axis=0)[:, np.newaxis]))
            modulation_angle[:, i] = self.wfs.thetaModulation

        return deltaI, modulation_angle

    def getDeltaIFactorsPerFrame(self, numFramesRemoved, D_amp):
        deltaI = np.zeros((len(numFramesRemoved), self.numBases))

        for i in range(self.numBases):
            # Start by making the flat (no distortion image)
            self.tel - self.atm
            self.tel * self.wfs
            cube_ref_no_remove = self.wfs.cam.cube

            # Get phase mask on atmosphere
            self.atm.OPD_no_pupil = self.bases[i, :, :] * D_amp
            self.atm.OPD = self.atm.OPD_no_pupil * self.tel.pupil
            # Apply atmosphere to telescope
            self.tel + self.atm
            # Propagate to wfs
            self.tel * self.wfs
            # Grab image
            cube_no_remove = self.wfs.cam.cube
            #Clear atmosphere
            self.tel - self.atm
            for j in range(len(numFramesRemoved)):
                removed_frames = calcEquidistantFrameIndices(numFramesRemoved[j], self.wfs.nTheta)
                deltaI[j,i] = calcDeltaI(cube_no_remove,cube_ref_no_remove, removed_frames)

        return deltaI


    def save2file(self):
        suffix = "mag" + str(self.param['magnitude']) + "_" + time.strftime("%Y%m%d-%H%M%S") + ".csv"
        header = ','.join(str(e) for e in self.numFramesRemoved)
        np.savetxt("a_err_" + suffix, self.a_err, delimiter=",", header=header)
        np.savetxt("a_err_noisy_" + suffix, self.a_err_noisy, delimiter=",", header=header)
        for j in range(len(self.numFramesRemoved)):
            name = "a_" + str(self.numFramesRemoved[j]) + "removed_" + suffix
            np.savetxt(name, self.a[:, j, :], delimiter=",")
            name = "a_est_" + str(self.numFramesRemoved[j]) + "removed_" + suffix
            np.savetxt(name, self.a_est[:, j, :], delimiter=",")
            name = "a_noisy" + str(self.numFramesRemoved[j]) + "removed_" + suffix
            np.savetxt(name, self.a_noisy[:, j, :], delimiter=",")
            name = "a_est_noisy" + str(self.numFramesRemoved[j]) + "removed_" + suffix
            np.savetxt(name, self.a_est_noisy[:, j, :], delimiter=",")

    def display(self):

        fig = plt.figure()
        plt.plot(self.numFramesRemoved, np.mean(self.a_err, axis=0), color='b', label="No noise")
        plt.plot(self.numFramesRemoved, np.mean(self.a_err_noisy, axis=0), color='r', label="With noise")
        plt.errorbar(self.numFramesRemoved, np.mean(self.a_err, axis=0), np.std(self.a_err, axis=0))
        plt.errorbar(self.numFramesRemoved, np.mean(self.a_err_noisy, axis=0), np.std(self.a_err_noisy, axis=0))

        sub = f"Mag={self.param['magnitude']}, Modulation={self.param['modulation']}, a_amp={self.a_amp}, D_amp={self.D_amp}, iterations={self.iterations}"
        plt.title("Error as a function of the number of removed frames \n" + sub)
        plt.xlabel("Number of removed frames per face")
        plt.ylabel("Normed error in m")
        plt.legend()
        plt.show(block=True)



def runReconstruction(traj):
    T = TRPWFS(traj)

    T.simulateReconstructionError(traj.D_amp, traj.D_amp, traj.nLoop)

    traj.f_add_result("a_err_noisy", T.a_err_noisy, comment="Error on the reconstruction of a")
    traj.f_add_result("a_est_noisy", T.a_est_noisy, comment="Estimation of a vector")
    traj.f_add_result("a_noisy", T.a_noisy, comment="The a vector")

def dict_to_trajectory(d, traj):
    for k,v in d.items():
        traj.f_add_parameter(k, v, comment=k)

if __name__ == "__main__":
    from trwfs.parameter_files.parameterFile_TR_PWFS_general import initializeParameterFile

    param = initializeParameterFile()

    param["magnitude"] = 0

    # Create an environment that handles running our simulation
    env = Environment(trajectory='run', filename='/mnt/home/usager/cars2019/Documents/DATA/trwfs/trwfs_recon_testing_pn.hdf5',
                      file_title='trwfs_recon_testinghdf5',
                      comment="Testing reconstruction",
                      large_overview_tables=True,
                      log_config='DEFAULT',
                      log_stdout=True,
                      overwrite_file=True)

    # Get the trajectory from the environment
    traj = env.trajectory

    dict_to_trajectory(param, traj)
    traj.f_add_parameter('recon_mode', "binary_opt", comment='The type of reconstruction to use')
    traj.f_add_parameter('D_amp', (30*1e-9), comment='Amplitude used for the interaction matrix')

    traj.f_explore(cartesian_product({'nTheta_user_defined': [48],
                                      "nModes": [50],
                                      "nLoop": [50],
                                      "recon_mode": ["normal", "binary_opt", "weighted"],
                                      'modulation': [5]}))


    env.run(runReconstruction)

    # Let's check that all runs are completed!
    assert traj.f_is_completed()

    # Finally disable logging and close all log-files
    env.disable_logging()

