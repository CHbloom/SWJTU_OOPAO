from trwfs.tools.PWFS_CMOS_NOISE_PROOF_may2022_tools import *
import pickle

from OOPAO.TRM_Pyramid          import TRM_Pyramid


class TRPWFS():

    def __init__(self, param, numBases, nTheta_user_defined=48, recon_mode="binary", display=False ):
        self.param = param
        # %% -----------------------     TELESCOPE   ----------------------------------
        # create the Telescope object
        self.tel = Telescope(resolution=param['resolution'], \
                        diameter=param['diameter'], \
                        samplingTime=param['samplingTime'], \
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

        # %% -----------------------     PYRAMID WFS   ----------------------------------
        if recon_mode == "binary":
            if (((nTheta_user_defined // 4) % 2) == 0):
                REMOVED_FRAMES_SETTINGS = np.array([0])
                frms = np.array(list(range((nTheta_user_defined // 4))))
                REMOVED_FRAMES_SETTINGS = np.append(REMOVED_FRAMES_SETTINGS, frms[frms % 2 == 1])  # [0,1,3,5,7,9,11]
            else:
                frms = np.array(list(range((nTheta_user_defined // 4))))
                REMOVED_FRAMES_SETTINGS = frms[frms % 2 == 0]

            temporal_weights_settings = np.ones((len(REMOVED_FRAMES_SETTINGS), nTheta_user_defined))
            for i in range(len(REMOVED_FRAMES_SETTINGS)):
                removed_frames = calcEquidistantFrameIndices(REMOVED_FRAMES_SETTINGS[i], nTheta_user_defined)
                temporal_weights_settings[i, removed_frames] = 0
        elif recon_mode == "weighted":
            pass




        self.wfs = TRM_Pyramid(nSubap=param['nSubaperture'], \
                      telescope=self.tel, \
                      modulation=param['modulation'], \
                      lightRatio=param['lightThreshold'], \
                      pupilSeparationRatio=param['pupilSeparationRatio'], \
                      calibModulation=param['calibrationModulation'], \
                      psfCentering=param['psfCentering'], \
                      edgePixel=param['edgePixel'], \
                      extraModulationFactor=param['extraModulationFactor'], \
                      postProcessing=param['postProcessing'],
                      nTheta_user_defined=nTheta_user_defined,
                      temporal_weights_settings=temporal_weights_settings)



        if isinstance(numBases, list):
            self.numBases = len(numBases)
            self.bases = generateBases(num=(np.max(numBases)+1), res=self.tel.resolution, baseType="KL", display=display, scale=False)
            self.bases = self.bases[(np.array(numBases)),:,:]
        else:
            self.numBases = numBases
            self.bases = generateBases(num=numBases, res=self.tel.resolution, baseType="KL", display=display, scale=False)


    def simulateReconstructionError(self, numFramesRemoved, D_amp, a_amp, iterations):
        self.a_amp = a_amp
        self.D_amp = D_amp
        self.iterations = iterations
        self.numFramesRemoved = numFramesRemoved
        self.a_err = np.zeros((iterations, len(numFramesRemoved)))
        self.a_est = np.zeros((iterations, len(numFramesRemoved), self.numBases))
        self.a = np.zeros((iterations, len(numFramesRemoved), self.numBases))

        self.a_err_noisy = np.zeros((iterations, len(numFramesRemoved)))
        self.a_est_noisy = np.zeros((iterations, len(numFramesRemoved), self.numBases))
        self.a_noisy = np.zeros((iterations, len(numFramesRemoved), self.numBases))

        self.a_err_noisy_custom = np.zeros((iterations))
        self.a_est_noisy_custom = np.zeros((iterations, self.numBases))
        self.a_noisy_custom = np.zeros((iterations, self.numBases))

        Dplus = [None] * len(numFramesRemoved)
        Dplus_custom = None

        # custom_remove_frames = np.zeros((self.numBases))
        # custom_remove_frames[0:4] = 9
        # custom_remove_frames[4:10] = 7
        # custom_remove_frames[10:16] = 5
        # custom_remove_frames[16:21] = 3
        # custom_remove_frames[21:27] = 1

        custom_remove_frames = np.array([11, 11, 9, 9, 9, 7, 7, 7, 7, 7, 7, 7, 7, 5, 5, 5, 5, 5, 5, 5, 5, 5, 3, 3, 3, 3, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])


        cube_ref = None
        for i in tqdm(range(iterations)):
            force_a = generateA(self.numBases, (200*1e-9))
            cube = None
            cube_noisy = None
            for j in range(len(numFramesRemoved)):
                removed_frames = calcEquidistantFrameIndices(numFramesRemoved[j], self.wfs.nTheta)
                self.a[i, j, :], self.a_est[i, j, :], self.a_err[i, j], Dplus[j], cube, cube_ref  = calcReconstructionError(self.atm, self.tel, self.wfs, self.bases, removed_frames,
                                                                                  D_amp=D_amp,
                                                                                  a_amp=a_amp,
                                                                                  display=False,
                                                                                  noise=False,
                                                                                  seed=i * 42,
                                                                                  force_a=force_a,
                                                                                  force_Dplus=Dplus[j],
                                                                                  forceCube=cube,
                                                                                  forceRefCube=cube_ref)
                self.a_noisy[i, j, :], self.a_est_noisy[i, j, :], self.a_err_noisy[i, j], Dplus[j], cube_noisy, cube_ref = calcReconstructionError(self.atm, self.tel, self.wfs,
                                                                                                    self.bases,
                                                                                                    removed_frames,
                                                                                                    D_amp=D_amp,
                                                                                                    a_amp=a_amp,
                                                                                                    display=False,
                                                                                                    noise=True,
                                                                                                    seed=i * 42,
                                                                                                    force_a=force_a,
                                                                                                    force_Dplus=Dplus[j],
                                                                                                    forceCube=cube_noisy,
                                                                                                    forceRefCube=cube_ref)
            self.a_noisy_custom[i, :], self.a_est_noisy_custom[i, :], self.a_err_noisy_custom[i], Dplus_custom = calcReconstructionErrorMultiD(self.atm, self.tel, self.wfs,
                                             self.bases,
                                             custom_remove_frames,
                                             D_amp=D_amp,
                                             a_amp=a_amp,
                                             display=False,
                                             noise=True,
                                             seed=i * 42,
                                             force_a=force_a,
                                             force_Dplus=Dplus_custom,
                                             forceCube=cube_noisy,
                                             forceRefCube=cube_ref)

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

if __name__ == "__main__":
    from parameter_files.parameterFile_CMOS_PWFS_aug2022 import initializeParameterFile

    nTheta_user_defined = 48

    if (((nTheta_user_defined // 4) % 2) == 0):
        REMOVED_FRAMES_SETTINGS = np.array([0])
        frms = np.array(list(range((nTheta_user_defined // 4))))
        REMOVED_FRAMES_SETTINGS = np.append(REMOVED_FRAMES_SETTINGS, frms[frms % 2 == 1])  # [0,1,3,5,7,9,11]
    else:
        frms = np.array(list(range((nTheta_user_defined // 4))))
        REMOVED_FRAMES_SETTINGS = frms[frms % 2 == 0]

    #REMOVED_FRAMES_SETTINGS = [0,1,3,5,7,9]

    param = initializeParameterFile()

    bases = 100
    a_err_bases = np.zeros((bases, len(REMOVED_FRAMES_SETTINGS)))
    a_err_noisy_bases = np.zeros((bases, len(REMOVED_FRAMES_SETTINGS)))
    a_err_noisy_custom = np.zeros((bases))

    a_err_bases_recon_best = np.zeros((bases ))
    a_err_noisy_bases_recon_best = np.zeros((bases ))

    a_err_bases_std = np.zeros((bases, len(REMOVED_FRAMES_SETTINGS)))
    a_err_noisy_bases_std = np.zeros((bases, len(REMOVED_FRAMES_SETTINGS)))

    #for i in range(len(bases)):
    T = TRPWFS(param, bases, nTheta_user_defined=nTheta_user_defined)
    T.simulateReconstructionError(REMOVED_FRAMES_SETTINGS, D_amp=(30* 1e-9), a_amp=(30* 1e-9), iterations=50)
    a_err_bases[:,:] = np.mean(np.abs(T.a_est - T.a), axis=0).T
    a_err_noisy_bases[:, :] = np.mean(np.abs(T.a_est_noisy - T.a_noisy), axis=0).T
    a_err_noisy_custom[:] = np.mean(np.abs(T.a_est_noisy_custom - T.a_noisy_custom), axis=0)
    a_err_bases_std[:, :] = np.std(np.abs(T.a_est - T.a), axis=0).T
    a_err_noisy_bases_std[:, :] = np.std(np.abs(T.a_est_noisy - T.a_noisy), axis=0).T

    np.nan_to_num(a_err_bases, nan=10000)
    np.nan_to_num(a_err_noisy_bases, nan=10000)
    np.nan_to_num(a_err_bases_std, nan=10000)
    np.nan_to_num(a_err_noisy_bases_std, nan=10000)

    for i in range(bases):
        a_err_bases_recon_best[i] = REMOVED_FRAMES_SETTINGS[np.argmin(a_err_bases[i,:])]
        a_err_noisy_bases_recon_best[i] = REMOVED_FRAMES_SETTINGS[np.argmin(a_err_noisy_bases[i, :])]


    saveObj = {"REMOVED_FRAMES_SETTINGS": REMOVED_FRAMES_SETTINGS,
               "Magnitudes": param["magnitude"],
               "a_err_bases": a_err_bases,
               "a_err_noisy_bases":a_err_noisy_bases,
               "a_err_bases_std":a_err_bases_std,
               "a_err_noisy_bases_std":a_err_noisy_bases_std,
               "iterations": T.iterations,
               "bases": bases,
               "D_amp": T.D_amp,
               "a_amp": T.a_amp
               }
    with open('data_bases.pickle', 'wb') as f:
        pickle.dump(saveObj, f)

    lines = []
    plt.ion()
    fig, (ax1,ax2) = plt.subplots(1,2, figsize=(10,10))
    for i in range(len(REMOVED_FRAMES_SETTINGS)):
        #plt.plot(list(range(bases)), a_err_bases[:,i], label=f"{REMOVED_FRAMES_SETTINGS[i]} removed")
        #plt.plot(list(range(bases)), a_err_noisy_bases[:,i], label=f"{REMOVED_FRAMES_SETTINGS[i]} removed - Noisy")
        #plt.errorbar(list(range(bases)), a_err_bases[:,i], a_err_bases_std[:,i], label=f"{REMOVED_FRAMES_SETTINGS[i]} removed")

        l = ax1.errorbar(list(range(bases)), a_err_noisy_bases[:,i], a_err_noisy_bases_std[:,i], label=f"{REMOVED_FRAMES_SETTINGS[i]} removed - Noisy")
        lines.append(l[0])

    l = ax1.plot(list(range(bases)), a_err_noisy_custom, label=f"custom removed - Noisy", ls='--', lw=4)
    lines.append(l)

    sub = f"Modulation={param['modulation']}, a_amp={T.a_amp:.2E}, D_amp={T.D_amp:.2E}"
    ax1.set_title("Error as a function of the base used \n" + sub)
    ax1.set_xlabel("Base (KL number)")
    ax1.set_ylabel("Normed error in m")
    leg = ax1.legend()

    #ax2.bar(list(range(bases)),a_err_bases_recon_best)
    ax2.bar(list(range(bases)),a_err_noisy_bases_recon_best)
    print(a_err_noisy_bases_recon_best)
    ax2.set_xlabel("Base (KL number)")
    ax2.set_ylabel("Best number of removed frames")

    plt.savefig('data.png')

    lined = {}  # Will map legend lines to original lines.
    for legline, origline in zip(leg.get_lines(), lines):
        legline.set_picker(True)  # Enable picking on the legend line.
        lined[legline] = origline


    def on_pick(event):
        print("test")
        # On the pick event, find the original line corresponding to the legend
        # proxy line, and toggle its visibility.
        legline = event.artist
        origline = lined[legline]
        visible = not origline.get_visible()
        origline.set_visible(visible)
        # Change the alpha on the line in the legend so we can see what lines
        # have been toggled.
        legline.set_alpha(1.0 if visible else 0.2)
        fig.canvas.draw()


    fig.canvas.mpl_connect('pick_event', on_pick)

    plt.pause(1)  # <---- add pause
    plt.show(block=True)
        #T.display()