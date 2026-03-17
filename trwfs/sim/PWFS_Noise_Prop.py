from trwfs.tools.PWFS_CMOS_NOISE_PROOF_may2022_tools import *
from trwfs.tools.TR_PWFS_Reconstruction import *
from trwfs.parameter_files.parameterFile_CMOS_PWFS_may2022 import initializeParameterFile


def simNoiseProp(numBases=50, nTheta_user_defined=48):

    if (((nTheta_user_defined // 4) % 2) == 0):
        REMOVED_FRAMES_SETTINGS = np.array([0])
        frms = np.array(list(range((nTheta_user_defined // 4))))
        REMOVED_FRAMES_SETTINGS = np.append(REMOVED_FRAMES_SETTINGS, frms[frms % 2 == 1])  # [0,1,3,5,7,9,11]
    else:
        frms = np.array(list(range((nTheta_user_defined // 4))))
        REMOVED_FRAMES_SETTINGS = frms[frms % 2 == 0]

    # REMOVED_FRAMES_SETTINGS = [0,1,3,5,7,9]

    param = initializeParameterFile()

    T = TRPWFS(param, numBases, nTheta_user_defined=nTheta_user_defined)

    B_diags = T.getErrorProgapation(REMOVED_FRAMES_SETTINGS, D_amp=(30 * 1e-9))

    deltaI = T.getDeltaIFactorsPerFrame(REMOVED_FRAMES_SETTINGS, D_amp=(30 * 1e-9))

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




if __name__ == "__main__":

    numBases = 40

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

    T = TRPWFS(param, numBases, nTheta_user_defined=nTheta_user_defined)

    B_diags = T.getErrorProgapation(REMOVED_FRAMES_SETTINGS, D_amp=(30*1e-9))

    deltaI = T.getDeltaIFactorsPerFrame(REMOVED_FRAMES_SETTINGS, D_amp=(30*1e-9))



    fig = plt.figure(4, figsize=(17, 10))
    ax = plt.subplot(4, 1, 1)

    ax.bar(np.arange(len(B_diags[0, :])), B_diags[0, :], label=f"{REMOVED_FRAMES_SETTINGS[0]}/face removed", color='k')
    for i in range(1,len(REMOVED_FRAMES_SETTINGS)):
        ax.bar(np.arange(len(B_diags[i,:])), B_diags[i,:], label=f"{REMOVED_FRAMES_SETTINGS[i]}/face removed")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax.set_xlabel("Mode")
    ax.set_ylabel("Noise factor (unitless?)")
    ax.set_title("Noise factor as a function of the mode")
    ax.legend()

    ax2 = plt.subplot(4, 1, 2)
    for i in range(1,len(REMOVED_FRAMES_SETTINGS)):
        ax2.plot((B_diags[i,:] / B_diags[0,:]) * 100, label=f"{REMOVED_FRAMES_SETTINGS[i]}/face removed")

    ax2.axhline(100, lw=3, linestyle='--', color='k')

    ax2.set_xlabel("Mode number")
    ax2.set_ylabel("Noise factor as % of no face removed")
    ax2.set_title('Percentage of the noise factor relative to \n no frames removed as a function of the mode')

    #ax2.legend()
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax3 = plt.subplot(4, 1, 3)

    ax3.plot( deltaI[0, :], label=f"{REMOVED_FRAMES_SETTINGS[0]}/face removed", color='k')
    for i in range(1, len(REMOVED_FRAMES_SETTINGS)):
        ax3.plot(deltaI[i, :], label=f"{REMOVED_FRAMES_SETTINGS[i]}/face removed")

    ax3.set_xlabel("Mode number")
    ax3.set_ylabel("Delta I")
    ax3.set_title('Delta I as function of the mode of the distortion applied')
    ax3.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax4 = plt.subplot(4, 1, 4)
    factor = np.zeros(B_diags.shape)
    for i in range(0, len(REMOVED_FRAMES_SETTINGS)):
        factor[i,:] = ((nTheta_user_defined-(REMOVED_FRAMES_SETTINGS[i]*4))/nTheta_user_defined) / B_diags[i, :]

    best_setting = [0]*numBases
    for j in range(numBases):
        best_setting[j] = REMOVED_FRAMES_SETTINGS[np.argmax(factor[:,j])]
    ax4.bar(list(range(numBases)), best_setting)
    print(best_setting)

    plt.show(block=True)

