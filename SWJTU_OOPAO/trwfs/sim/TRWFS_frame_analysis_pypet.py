import numpy as np

from trwfs.tools.PWFS_CMOS_NOISE_PROOF_may2022_tools import *
from trwfs.tools.TR_PWFS_Reconstruction import *
from trwfs.parameter_files.parameterFile_CMOS_PWFS_may2022 import initializeParameterFile
from pypet import Environment, cartesian_product



def run(traj):
    param = traj.parameters.f_to_dict(fast_access=True, short_names=True)
    # if (((param["nTheta_user_defined"] // 4) % 2) == 0):
    #     REMOVED_FRAMES_SETTINGS = np.array([0])
    #     frms = np.array(list(range((param["nTheta_user_defined"] // 4))))
    #     REMOVED_FRAMES_SETTINGS = np.append(REMOVED_FRAMES_SETTINGS, frms[frms % 2 == 1])  # [0,1,3,5,7,9,11]
    # else:
    #     frms = np.array(list(range((param["nTheta_user_defined"] // 4))))
    #     REMOVED_FRAMES_SETTINGS = frms[frms % 2 == 0]


    T = TRPWFS(param, param["numBases"], nTheta_user_defined=param["nTheta_user_defined"])

    #B_diags = T.getErrorProgapation(REMOVED_FRAMES_SETTINGS, D_amp=param["D_amp"])

    deltaI_per_frame, modulation_angle = T.getDeltaIPerFrame(D_amp=param["D_amp"])

    #traj.f_add_result("removed_frames_settings", REMOVED_FRAMES_SETTINGS, comment="The different removed frames settings")
    #traj.f_add_result("b_diag", B_diags, comment="Diagonalisation of the pseudo-inverse of the interaction matrix. Represents the sensitivity to noise.")
    traj.f_add_result("deltai_per_frame", deltaI_per_frame, comment="Delta I information for each frame, for all bases")
    traj.f_add_result("modulation_angle", modulation_angle, comment="Angle of the modulation, corresponding to deltai_per_frame")



def view(traj, modulation, nTheta_user_defined, numBases, D_amp, maxBases=None):
    filter_function = lambda modulation_l, nTheta_user_defined_l, numBases_l, D_amp_l: modulation_l == modulation and \
                                                                                       nTheta_user_defined_l == nTheta_user_defined and \
                                                                                       numBases_l == numBases and \
                                                                                       D_amp_l == D_amp
    idx_iterator = traj.f_find_idx(['parameters.modulation', 'parameters.nTheta_user_defined', 'parameters.numBases', 'parameters.D_amp'], filter_function)
    for idx in idx_iterator:
        traj.v_idx = idx

    if maxBases is None:
        maxBases = numBases
    side = np.int(np.ceil(np.sqrt(maxBases)))
    fig, axs = plt.subplots(side,side) #, subplot_kw={'projection': 'polar'}
    deltaI = traj.crun.deltai_per_frame
    for base in range(maxBases):
        ax = axs[base//side, base%side]

        theta = (np.pi*2) - traj.crun.modulation_angle[:,base]
        ax.bar(theta, deltaI[:,base], width=(2*np.pi/theta.shape[0]), edgecolor="k")

    fig2, axs2 = plt.subplots(side, side)
    for base in range(maxBases):
        ax2 = axs2[base//side, base%side]
        theta = (np.pi*2) - traj.crun.modulation_angle[:,base]
        halfpoint = theta.shape[0]//2
        ax2.plot(np.abs(np.fft.fftshift(np.fft.ifft(deltaI[0:halfpoint,base],n=nTheta_user_defined))))


def dict_to_trajectory(d, traj):
    for k,v in d.items():
        traj.f_add_parameter(k, v, comment=k)


if __name__ == "__main__":
    from trwfs.parameter_files.parameterFile_CMOS_PWFS_aug2022_3 import initializeParameterFile

    param = initializeParameterFile()
    param["magnitude"] = 0

    # Create an environment that handles running our simulation
    env = Environment(trajectory='run', filename='/home/cars2019/DATA/HDF/trwfs_frame_analysis_testing_migr.hdf5',
                      file_title='trwfs_frame_analysis',
                      comment='Getting the information per frame, high frame count',
                      large_overview_tables=True,
                      log_config='DEFAULT',
                      log_stdout=True,
                      overwrite_file=True)

    # Get the trajectory from the environment
    traj = env.trajectory

    dict_to_trajectory(param, traj)
    traj.f_add_parameter('numBases', 9, comment='Number of KL bases to use')
    traj.f_add_parameter('D_amp', (1*1e-9), comment='Amplitude used for the interaction matrix')
    #traj.f_add_parameter('nTheta_user_defined', 48, comment='Number of frames per modulation')

    traj.f_explore(cartesian_product({'nTheta_user_defined': [48*100],
                                      'modulation': [5]}))


    env.run(run)

    # Let's check that all runs are completed!
    assert traj.f_is_completed()

    # Finally disable logging and close all log-files
    env.disable_logging()



# if __name__ == "__main__":
#
#     numBases = 50
#
#     nTheta_user_defined = 48*100
#
#     if (((nTheta_user_defined // 4) % 2) == 0):
#         REMOVED_FRAMES_SETTINGS = np.array([0])
#         frms = np.array(list(range((nTheta_user_defined // 4))))
#         REMOVED_FRAMES_SETTINGS = np.append(REMOVED_FRAMES_SETTINGS, frms[frms % 2 == 1])  # [0,1,3,5,7,9,11]
#     else:
#         frms = np.array(list(range((nTheta_user_defined // 4))))
#         REMOVED_FRAMES_SETTINGS = frms[frms % 2 == 0]
#
#     #REMOVED_FRAMES_SETTINGS = [0,1,3,5,7,9]
#
#     param = initializeParameterFile()
#
#     T = TRPWFS(param, numBases, nTheta_user_defined=nTheta_user_defined)
#
#     B_diags = T.getErrorProgapation(REMOVED_FRAMES_SETTINGS, D_amp=(30*1e-9))
#
#     deltaI = T.getDeltaIPerFrame(D_amp=(30*1e-9))
#
#
#
#     fig = plt.figure(4, figsize=(17, 10))
#     ax = plt.subplot(4, 1, 1)
#
#     ax.bar(np.arange(len(B_diags[0, :])), B_diags[0, :], label=f"{REMOVED_FRAMES_SETTINGS[0]}/face removed", color='k')
#     for i in range(1,len(REMOVED_FRAMES_SETTINGS)):
#         ax.bar(np.arange(len(B_diags[i,:])), B_diags[i,:], label=f"{REMOVED_FRAMES_SETTINGS[i]}/face removed")
#     ax.xaxis.set_major_locator(MaxNLocator(integer=True))
#
#     ax.set_xlabel("Mode")
#     ax.set_ylabel("Noise factor (unitless?)")
#     ax.set_title("Noise factor as a function of the mode")
#     ax.legend()
#
#     ax2 = plt.subplot(4, 1, 2)
#     for i in range(1,len(REMOVED_FRAMES_SETTINGS)):
#         ax2.plot((B_diags[i,:] / B_diags[0,:]) * 100, label=f"{REMOVED_FRAMES_SETTINGS[i]}/face removed")
#
#     ax2.axhline(100, lw=3, linestyle='--', color='k')
#
#     ax2.set_xlabel("Mode number")
#     ax2.set_ylabel("Noise factor as % of no face removed")
#     ax2.set_title('Percentage of the noise factor relative to \n no frames removed as a function of the mode')
#
#     #ax2.legend()
#     ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
#
#     ax3 = plt.subplot(4, 1, 3)
#
#     ax3.plot( deltaI[0, :], label=f"{REMOVED_FRAMES_SETTINGS[0]}/face removed", color='k')
#     for i in range(1, len(REMOVED_FRAMES_SETTINGS)):
#         ax3.plot(deltaI[i, :], label=f"{REMOVED_FRAMES_SETTINGS[i]}/face removed")
#
#     ax3.set_xlabel("Mode number")
#     ax3.set_ylabel("Delta I")
#     ax3.set_title('Delta I as function of the mode of the distortion applied')
#     ax3.xaxis.set_major_locator(MaxNLocator(integer=True))
#
#     ax4 = plt.subplot(4, 1, 4)
#     factor = np.zeros(B_diags.shape)
#     for i in range(0, len(REMOVED_FRAMES_SETTINGS)):
#         factor[i,:] = ((nTheta_user_defined-(REMOVED_FRAMES_SETTINGS[i]*4))/nTheta_user_defined) / B_diags[i, :]
#
#     best_setting = [0]*numBases
#     for j in range(numBases):
#         best_setting[j] = REMOVED_FRAMES_SETTINGS[np.argmax(factor[:,j])]
#     ax4.bar(list(range(numBases)), best_setting)
#     print(best_setting)
#
#     plt.show(block=True)

