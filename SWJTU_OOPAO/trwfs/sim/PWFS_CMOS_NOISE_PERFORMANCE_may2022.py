import matplotlib.pyplot as plt

from PWFS_CMOS_NOISE_PROOF_may2022_tools import *



#%% -----------------------     read parameter file   ----------------------------------
from parameter_files.parameterFile_CMOS_PWFS_may2022 import initializeParameterFile
param = initializeParameterFile()
#%% -----------------------     TELESCOPE   ----------------------------------
# create the Telescope object
tel = Telescope(resolution          = param['resolution'],\
                diameter            = param['diameter'],\
                samplingTime        = param['samplingTime'],\
                centralObstruction  = param['centralObstruction'])

#%% -----------------------     NGS   ----------------------------------
# create the Source object
ngs=Source(optBand   = param['opticalBand'],\
           magnitude = param['magnitude'])

# combine the NGS to the telescope using '*' operator:
ngs*tel

tel.computePSF(zeroPaddingFactor = 6)

#%% -----------------------     ATMOSPHERE   ---------------------------------

"Dummy atmosphere to be able to fit whatever aberration we want"
atm = dummy_atm(tel)


#%% -----------------------     PYRAMID WFS   ----------------------------------
wfs = Pyramid(nSubap                = param['nSubaperture'],\
              telescope             = tel,\
              modulation            = param['modulation'],\
              lightRatio            = param['lightThreshold'],\
              pupilSeparationRatio  = param['pupilSeparationRatio'],\
              calibModulation       = param['calibrationModulation'],\
              psfCentering          = param['psfCentering'],\
              edgePixel             = param['edgePixel'],\
              extraModulationFactor = param['extraModulationFactor'],\
              postProcessing        = param['postProcessing'])


NUMBER_OF_BASES = 9

bases = generateBases(num=NUMBER_OF_BASES, res=tel.resolution, baseType="KL", display=False, scale=False)
#bases = bases[14:,]
#plt.show(block=True)


removed_frames = calcEquidistantFrameIndices(5,wfs.nTheta)

#calcPerformance(atm, tel, wfs, bases, removed_frames=removed_frames, display=False)
#plt.show(block=True)


D_amp = (30 * 1e-9)
a_amp = (30 * 1e-9)

NUM_OF_ITERATIONS = 20
REMOVED_FRAMES_SETTINGS = [0,1,3,5,7,9]

a_err = np.zeros((NUM_OF_ITERATIONS, len(REMOVED_FRAMES_SETTINGS)))
a_est = np.zeros((NUM_OF_ITERATIONS,  len(REMOVED_FRAMES_SETTINGS), NUMBER_OF_BASES))
a = np.zeros((NUM_OF_ITERATIONS,  len(REMOVED_FRAMES_SETTINGS), NUMBER_OF_BASES))

a_err_noisy = np.zeros((NUM_OF_ITERATIONS, len(REMOVED_FRAMES_SETTINGS)))
a_est_noisy = np.zeros((NUM_OF_ITERATIONS,  len(REMOVED_FRAMES_SETTINGS), NUMBER_OF_BASES))
a_noisy = np.zeros((NUM_OF_ITERATIONS,  len(REMOVED_FRAMES_SETTINGS), NUMBER_OF_BASES))

Dplus = [None]*len(REMOVED_FRAMES_SETTINGS)
for i in tqdm(range(NUM_OF_ITERATIONS)):
    force_a = (np.random.randn(NUMBER_OF_BASES, ))
    for j in range(len(REMOVED_FRAMES_SETTINGS)):
        removed_frames = calcEquidistantFrameIndices(REMOVED_FRAMES_SETTINGS[j], wfs.nTheta)

        a[i,j,:], a_est[i, j, :], a_err[i,j], Dplus[j] = calcReconstructionError(atm, tel, wfs, bases, removed_frames,
                                                                       D_amp=D_amp,
                                                                       a_amp=a_amp,
                                                                       display=False,
                                                                       noise=False,
                                                                       seed=i*42,
                                                                       force_a=force_a,
                                                                       force_Dplus=Dplus[j])
        a_noisy[i,j,:], a_est_noisy[i, j, :], a_err_noisy[i,j], Dplus[j] = calcReconstructionError(atm, tel, wfs, bases, removed_frames,
                                                                                         D_amp=D_amp,
                                                                                         a_amp=a_amp,
                                                                                         display=False,
                                                                                         noise=True,
                                                                                         seed=i*42,
                                                                                         force_a=force_a,
                                                                                         force_Dplus=Dplus[j])

suffix = "mag" +str(param['magnitude']) +"_"+time.strftime("%Y%m%d-%H%M%S") + ".csv"
header = ','.join(str(e) for e in REMOVED_FRAMES_SETTINGS)
np.savetxt("a_err_"+suffix, a_err, delimiter=",", header=header)
np.savetxt("a_err_noisy_"+suffix, a_err_noisy, delimiter=",", header=header)
for j in range(len(REMOVED_FRAMES_SETTINGS)):
    name = "a_"+str(REMOVED_FRAMES_SETTINGS[j])+"removed_" + suffix
    np.savetxt(name, a[:,j,:], delimiter=",")
    name = "a_est_" + str(REMOVED_FRAMES_SETTINGS[j]) + "removed_" + suffix
    np.savetxt(name, a_est[:,j,:], delimiter=",")
    name = "a_noisy" + str(REMOVED_FRAMES_SETTINGS[j]) + "removed_" + suffix
    np.savetxt(name, a_noisy[:, j, :], delimiter=",")
    name = "a_est_noisy" + str(REMOVED_FRAMES_SETTINGS[j]) + "removed_" + suffix
    np.savetxt(name, a_est_noisy[:, j, :], delimiter=",")

t = np.mean(a_err, axis=0)
t2 = np.mean(a_err, axis=1)
fig = plt.figure()
# plt.plot(REMOVED_FRAMES_SETTINGS, np.mean(a_err, axis=0), color='b',label="No noise")
# plt.plot(REMOVED_FRAMES_SETTINGS, np.mean(a_err_noisy, axis=0), color='r', label="With noise")
plt.errorbar(REMOVED_FRAMES_SETTINGS, np.mean(a_err, axis=0), np.std(a_err, axis=0), label="No noise")
plt.errorbar(REMOVED_FRAMES_SETTINGS, np.mean(a_err_noisy, axis=0), np.std(a_err_noisy, axis=0), label="With noise")

sub = f"Mag={param['magnitude']}, Modulation={param['modulation']}, a_amp={a_amp}, D_amp={D_amp}"
plt.title("Error as a function of the number of removed frames \n" + sub )
plt.xlabel("Number of removed frames per face")
plt.ylabel("Normed error in m")
plt.legend()
plt.show(block=True)

"""Now we use random combinations of distortions and compare to see if removing frames benefits the reconstruction"""
'''
fig,_axs = plt.subplots(2,3, figsize=(10,10))
axs = _axs.flatten()

D_0 = generateInteractionMatrix(atm, tel, wfs, bases, removed_frames=[])
D_0 = D_0
D_plus_0 = inv(D_0.T @ D_0) @ D_0.T
D_adj = generateInteractionMatrix(atm, tel, wfs, bases, removed_frames=removed_frames)
D_adj= D_adj
D_plus_adj = inv(D_adj.T @ D_adj) @ D_adj.T

"""Random 'a' vector"""
# Perfer gaussian distribution here
a = (np.random.randn(9,)) # [-1,1]
a = a * 30 * (1e-9) # scale

"""The distortion associated to the 'a' vector"""
Phi_distortion = np.zeros(bases.shape)
for i in range(len(a)):
    Phi_distortion[i,:,:] = a[i] * bases[i,:,:]

Phi_distortion = np.sum(Phi_distortion, axis=0)

Phi_distortion_pupil  = (Phi_distortion) * tel.pupil

im0 = axs[0].imshow(Phi_distortion_pupil)
fig.colorbar(im0, ax=axs[0])
axs[0].set_title("Induced distortion")

im0 = axs[1].imshow(tel.src.phase)
fig.colorbar(im0, ax=axs[1])
axs[1].set_title("Telescope phase map")

im1 = axs[2].imshow(wfs.cam.frame)
fig.colorbar(im1, ax=axs[2])
axs[2].set_title("Reference pyramid image")

"""Calc images with distortions"""
P_0_img = getPyramidImageWithDistortion(atm,tel,wfs,Phi_distortion,removed_frames=[], noise=False, display=True, seed=0)
P_adj_img = getPyramidImageWithDistortion(atm,tel,wfs,Phi_distortion,removed_frames=removed_frames, noise=False, display=True, seed=0)



im1 = axs[3].imshow(P_0_img)
fig.colorbar(im1, ax=axs[3])
axs[3].set_title("$\Delta I$ without removed frames")

im2 = axs[4].imshow(P_adj_img)
fig.colorbar(im2, ax=axs[4])
axs[4].set_title("$\Delta I$ with removed frames")

P_0 = P_0_img.flatten()
P_adj = P_adj_img.flatten()

a_0_est = D_plus_0@P_0
a_adj_est = D_plus_adj@P_adj

a_0_err = np.sqrt(np.sum((a_0_est-a)**2))
a_adj_err = np.sqrt(np.sum((a_adj_est-a)**2))

print(a-a_0_est)

print(a-a_adj_est)

print(a_0_err)
print(a_adj_err)

plt.show(block=True)


_,_, err_0 = calcReconstructionError(atm, tel, wfs, bases, [],
                                   D_amp=30 * (1e-9),
                                   a_amp=30 * (1e-9),
                                   display=False,
                                   noise=False,
                                   seed=0)

_,_, err_5 = calcReconstructionError(atm, tel, wfs, bases, removed_frames,
                                   D_amp=30 * (1e-9),
                                   a_amp=30 * (1e-9),
                                   display=False,
                                   noise=False,
                                   seed=0)

print(err_0)
print(err_5)
'''