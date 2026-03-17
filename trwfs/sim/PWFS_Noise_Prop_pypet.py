from trwfs.tools.PWFS_CMOS_NOISE_PROOF_may2022_tools import *
from trwfs.tools.TR_PWFS_Reconstruction import *
from trwfs.parameter_files.parameterFile_CMOS_PWFS_may2022 import initializeParameterFile
from pypet import Environment, cartesian_product


def simNoiseProp(traj):
    param = traj.parameters.f_to_dict(fast_access=True, short_names=True)
    nTheta_user_defined = param["nTheta_user_defined"]
    numBases = param['nModes']
    D_amp = param["D_amp"]
    mode = param["mode"]

    # create the Telescope object
    tel = Telescope(resolution=param['resolution'], \
                    diameter=param['diameter'], \
                    samplingTime=param['samplingTime'], \
                    centralObstruction=param['centralObstruction'])

    # %% -----------------------     NGS   ----------------------------------
    # create the Source object
    ngs = Source(optBand=param['opticalBand'], \
                 magnitude=param['magnitude'])

    # combine the NGS to the telescope using '*' operator:
    ngs * tel

    tel.computePSF(zeroPaddingFactor=6)

    # %% -----------------------     ATMOSPHERE   ---------------------------------

    "Dummy atmosphere to be able to fit whatever aberration we want"
    atm = dummy_atm(tel)
    bases = generateBases(num=param["nModes"], res=tel.resolution, baseType="KL", display=False,
                          scale=False)

    wfs = TRM_Pyramid(nSubap=param['nSubaperture'], \
                      telescope=tel, \
                      modulation=param['modulation'], \
                      lightRatio=param['lightThreshold'], \
                      pupilSeparationRatio=param['pupilSeparationRatio'], \
                      calibModulation=param['calibrationModulation'], \
                      psfCentering=param['psfCentering'], \
                      edgePixel=param['edgePixel'], \
                      extraModulationFactor=param['extraModulationFactor'], \
                      postProcessing=param['postProcessing'],
                      nTheta_user_defined=param["nTheta_user_defined"],
                      temporal_weights_settings=None)
    frame_mask_setting = np.ones((nTheta_user_defined, numBases))
    if (mode == "normal"):
        B_diags, frame_mask_settings = calcBDiag(atm=atm, tel=tel, wfs=wfs, bases=bases, removed_frames=None, D_amp=D_amp, mode=None)
        traj.f_add_result("frame_mask_settings", frame_mask_settings, comment="Setting for each frame of the modulation")
        traj.f_add_result("B_diags", B_diags, comment="The diagonal of the B matrix, represents the impact of noise. Lower is better.")

    elif (mode == "binary_optimal"):
        B_diags, frame_mask_settings = calcBDiag(atm=atm, tel=tel, wfs=wfs, bases=bases, removed_frames=[], D_amp=D_amp, mode="binary")
        traj.f_add_result("frame_mask_settings", frame_mask_settings,
                          comment="Setting for each frame of the modulation")
        traj.f_add_result("B_diags", B_diags,
                          comment="The diagonal of the B matrix, represents the impact of noise. Lower is better.")

    elif (mode == "binary"):
        if (((nTheta_user_defined // 4) % 2) == 0):
            REMOVED_FRAMES_SETTINGS = np.array([0])
            frms = np.array(list(range((nTheta_user_defined // 4))))
            REMOVED_FRAMES_SETTINGS = np.append(REMOVED_FRAMES_SETTINGS, frms[frms % 2 == 1])  # [0,1,3,5,7,9,11]
        else:
            frms = np.array(list(range((nTheta_user_defined // 4))))
            REMOVED_FRAMES_SETTINGS = frms[frms % 2 == 0]

        B_diags = np.zeros((len(REMOVED_FRAMES_SETTINGS), numBases))
        frame_mask_settings = np.zeros((len(REMOVED_FRAMES_SETTINGS), nTheta_user_defined, numBases))
        for i in range(len(REMOVED_FRAMES_SETTINGS)):
            removed_frames = calcEquidistantFrameIndices(int(REMOVED_FRAMES_SETTINGS[i]), wfs.nTheta)
            B_diags[i, :], frame_mask_settings[i, :, :] = calcBDiag(atm=atm, tel=tel, wfs=wfs, bases=bases, removed_frames=removed_frames, D_amp=D_amp, mode=None)

        traj.f_add_result("frame_mask_settings", frame_mask_settings,
                          comment="Setting for each frame of the modulation")
        traj.f_add_result("B_diags", B_diags,
                          comment="The diagonal of the B matrix, represents the impact of noise. Lower is better.")

    elif (mode == "weighted"):
        B_diags, frame_mask_settings = calcBDiag(atm=atm, tel=tel, wfs=wfs, bases=bases, removed_frames=[], D_amp=D_amp, mode="weighted")
        traj.f_add_result("frame_mask_settings", frame_mask_settings,
                          comment="Setting for each frame of the modulation")
        traj.f_add_result("B_diags", B_diags,
                          comment="The diagonal of the B matrix, represents the impact of noise. Lower is better.")
    else:
        print("Unrecognized mode. Need 'binary' or 'weighted'")
        return -1



def simNoiseProp_view():
    fig = plt.figure(4, figsize=(17, 10))
    ax = plt.subplot(4, 1, 1)

    ax.bar(np.arange(len(B_diags[0, :])), B_diags[0, :], label=f"{REMOVED_FRAMES_SETTINGS[0]}/face removed", color='k')
    for i in range(1, len(REMOVED_FRAMES_SETTINGS)):
        ax.bar(np.arange(len(B_diags[i, :])), B_diags[i, :], label=f"{REMOVED_FRAMES_SETTINGS[i]}/face removed")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax.set_xlabel("Mode")
    ax.set_ylabel("Noise factor (unitless?)")
    ax.set_title("Noise factor as a function of the mode")
    ax.legend()

    ax2 = plt.subplot(4, 1, 2)
    for i in range(1, len(REMOVED_FRAMES_SETTINGS)):
        ax2.plot((B_diags[i, :] / B_diags[0, :]) * 100, label=f"{REMOVED_FRAMES_SETTINGS[i]}/face removed")

    ax2.axhline(100, lw=3, linestyle='--', color='k')

    ax2.set_xlabel("Mode number")
    ax2.set_ylabel("Noise factor as % of no face removed")
    ax2.set_title('Percentage of the noise factor relative to \n no frames removed as a function of the mode')

    # ax2.legend()
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax3 = plt.subplot(4, 1, 3)

    ax3.plot(deltaI[0, :], label=f"{REMOVED_FRAMES_SETTINGS[0]}/face removed", color='k')
    for i in range(1, len(REMOVED_FRAMES_SETTINGS)):
        ax3.plot(deltaI[i, :], label=f"{REMOVED_FRAMES_SETTINGS[i]}/face removed")

    ax3.set_xlabel("Mode number")
    ax3.set_ylabel("Delta I")
    ax3.set_title('Delta I as function of the mode of the distortion applied')
    ax3.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax4 = plt.subplot(4, 1, 4)
    factor = np.zeros(B_diags.shape)
    for i in range(0, len(REMOVED_FRAMES_SETTINGS)):
        factor[i, :] = ((nTheta_user_defined - (REMOVED_FRAMES_SETTINGS[i] * 4)) / nTheta_user_defined) / B_diags[i, :]

    best_setting = [0] * numBases
    for j in range(numBases):
        best_setting[j] = REMOVED_FRAMES_SETTINGS[np.argmax(factor[:, j])]
    ax4.bar(list(range(numBases)), best_setting)
    print(best_setting)




def dict_to_trajectory(d, traj):
    for k,v in d.items():
        traj.f_add_parameter(k, v, comment=k)


if __name__ == "__main__":
    from trwfs.parameter_files.parameterFile_TR_PWFS_general import initializeParameterFile

    param = initializeParameterFile()
    param["magnitude"] = 0

    # Create an environment that handles running our simulation
    env = Environment(trajectory='run', filename='/home/cars2019/DATA/HDF/testing_noise_prop.hdf5',
                      file_title='testing_noise_prop',
                      comment='Testing noise prop',
                      large_overview_tables=True,
                      log_config='DEFAULT',
                      log_stdout=True,
                      overwrite_file=True)

    # Get the trajectory from the environment
    traj = env.trajectory

    dict_to_trajectory(param, traj)
    traj.f_add_parameter('mode', "binary", comment='In what mode to operate the PWFS')
    traj.f_add_parameter('D_amp', (1*1e-9), comment='Amplitude of the distortion when calculating the interaction matrix')

    traj.f_explore(cartesian_product({'nModes': [50],
                                      'mode': ["binary", ]}))


    env.run(simNoiseProp)

    # Let's check that all runs are completed!
    assert traj.f_is_completed()

    # Finally disable logging and close all log-files
    env.disable_logging()
