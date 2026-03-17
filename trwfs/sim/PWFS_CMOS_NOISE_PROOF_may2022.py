# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 10:51:32 2020

@author: cheritie
"""
import copy

import aotools.functions.zernike
from joblib import Parallel, delayed
import imageio, os

import PWFS_CMOS_NOISE_PROOF_may2022_tools

# commom modules
#import matplotlib.backends.backend_qt5agg
from matplotlib.ticker import MaxNLocator
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('Qt5Agg')
import numpy             as np
from numpy.linalg import inv
import time
plt.ion()
import __load__psim
__load__psim.load_psim()

from AO_modules.Atmosphere       import Atmosphere
from AO_modules.Pyramid          import Pyramid
from AO_modules.DeformableMirror import DeformableMirror
from AO_modules.MisRegistration  import MisRegistration
from AO_modules.Telescope        import Telescope
from AO_modules.Source           import Source
# calibration modules 
from AO_modules.calibration.compute_KL_modal_basis import compute_M2C
from AO_modules.calibration.ao_calibration import ao_calibration
# display modules
from AO_modules.tools.displayTools           import displayMap

#%% -----------------------     read parameter file   ----------------------------------
from parameter_files.parameterFile_CMOS_PWFS_may2022 import initializeParameterFile
param = initializeParameterFile()



#fig_init = plt.figure(1)
pause = False


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
# ax1 = plt.subplot(3,2,1)
# telescope_view = ax1.imshow(np.log10(np.abs(tel.PSF)),extent = [tel.xPSF_arcsec[0],tel.xPSF_arcsec[1],tel.xPSF_arcsec[0],tel.xPSF_arcsec[1]])
# telescope_view.set_clim(-1,3)
# ax1.set_xlabel('[Arcsec]')
# ax1.set_ylabel('[Arcsec]')
# plt.colorbar(telescope_view)
#%% -----------------------     ATMOSPHERE   ----------------------------------

# # create the Atmosphere object
# atm=Atmosphere(telescope     = tel,\
#                r0            = param['r0'],\
#                L0            = param['L0'],\
#                windSpeed     = param['windSpeed'],\
#                fractionalR0  = param['fractionnalR0'],\
#                windDirection = param['windDirection'],\
#                altitude      = param['altitude'])
# # initialize atmosphere
# atm.initializeAtmosphere(tel)
#
# atm.update()

class temp_atm():
    def __init__(self, tel):
        self.OPD = None
        self.OPD_no_pupil = None
        self.wavelength = 500*1e-9
        self.tel = tel

    def __mul__(self,obj):
        obj.OPD=self.OPD
        obj.OPD_no_pupil=self.OPD_no_pupil
        obj.isPaired=True
        return obj

    def update(self):
        # P = np.zeros([self.tel.resolution, self.tel.resolution])
        # for i_layer in range(self.nLayer):
        #     tmpLayer = getattr(self, 'layer_' + str(i_layer + 1))
        #
        #     P += tmpLayer.phase * np.sqrt(self.fractionalR0[i_layer])

        # self.OPD_no_pupil = 1 * self.wavelength / 2 / np.pi
        # self.OPD = self.OPD_no_pupil * self.tel.pupil

        if self.tel.isPaired:
            self * self.tel

#Zernike Phase maskp
#phaseMask = aotools.functions.zernike.phaseFromZernikes([aotools.zernike_noll(4, 120)], 120)
# phaseMask_o,_,_,_ = aotools.functions.karhunenLoeve.make_kl(150, 120, stf='kolmogorov')
# phaseMask = phaseMask_o[6,:,:]



# Fourier Phase mask
def fourierPhaseMask(wavelengthPix, angleRad, size=120):
    x = np.arange(-(size/2), (size/2), 1)
    X, Y = np.meshgrid(x, x)

    phaseMask = np.sin(
        2*np.pi*(X*np.cos(angleRad) + Y*np.sin(angleRad)) / wavelengthPix
    )
    return phaseMask

def generateFourierBases(bases, resolution):
    """
    Generate the X first fourier bases with the given resolution. Where X is the bases argument.
    The bases are at 0 and 90deg angles.
    Parameters
    ----------
    bases : Number of bases to generate
    resolution : width and height of the result image

    Returns
    -------
    fourierBases : (bases*2, resolution, resolution)
    """
    fourierBases = np.zeros((bases*2,resolution,resolution))
    for i in range(bases):
        j = i*2
        fourierBases[  j, :, :] = fourierPhaseMask(resolution // (i+1),       0)  # Horizontal
        fourierBases[j+1, :, :] = fourierPhaseMask(resolution // (i+1), np.pi/2)  # Vertical

    return fourierBases

def displayBases(bases):
    numBases = bases.shape[0]
    side = int(np.ceil(np.sqrt(numBases)))
    fig, ax_i = plt.subplots(side, side, figsize=(10, 7))
    fig.suptitle('Bases used for the interaction matrix', fontsize=18)
    for x in range(side):
        for y in range(side):
            i = (x * side) + y
            ax_i[x, y].axis('off')
            if i < numBases:
                ax_i[x, y].imshow(bases[i, :, :])
            else:
                ax_i[x, y].set_visible(False)

    plt.show(block=True)

def generateBases(num, res, baseType, display=True):
    if baseType == "Fourier":
        bases = generateFourierBases(num, res)
    elif baseType == "KL":
        bases,_,_,_ = aotools.functions.karhunenLoeve.make_kl(num, res, stf='kolmogorov')
    else:
        print("Unrecognized base type. Choices are: Fourier, KL")
        print("Using KL by default")
        bases,_,_,_ = aotools.functions.karhunenLoeve.make_kl(num, res, stf='kolmogorov')
    if display:
        displayBases(bases)
    return bases

bases = generateBases(num=100, res=tel.resolution, baseType="KL", display=True)




#displayBases(bases)


def generateInteractionMatrix(atm, tel, wfs, bases, removed_frames=None):

    if removed_frames is None:
        removed_frames = []

    D = np.zeros((wfs.cam.resolution*wfs.cam.resolution, bases.shape[0]))

    # Start by making the flat (no distortion image)
    tel-atm
    tel*wfs
    cube_ref = wfs.cam.cube
    cube_ref = np.delete(cube_ref, removed_frames, axis=0)
    frame_ref = wfs.cam.frame



    # Iterate through the bases
    for b in range(bases.shape[0]):
        # Get phase mask on atmosphere
        atm.OPD_no_pupil = bases[b,:,:] * atm.wavelength / 2 / np.pi
        atm.OPD = atm.OPD_no_pupil * tel.pupil
        # Apply atmosphere to telescope
        tel+atm
        # Propagate to wfs
        tel*wfs
        # Grab image
        cube = wfs.cam.cube
        cube = np.delete(cube, removed_frames, axis=0)


        flat_ref = np.sum(cube_ref, axis=0)
        flat = np.sum(cube, axis=0)
        t  = np.sum(flat)
        t_ref = np.sum(flat_ref)
        I = (flat/np.sum(flat)) - (flat_ref/np.sum(flat_ref))
        I_flat = I.flatten()
        I_flat_normed = I_flat/np.linalg.norm(I_flat)
        D[:, b] = I_flat

        #Remove distortion
        tel - atm
    return D

def generateInteractionMatrixParalel(atm, tel, wfs, bases, removed_frames=None):

    if removed_frames is None:
        removed_frames = []

    D = np.zeros((wfs.cam.resolution*wfs.cam.resolution, bases.shape[0]))

    # Start by making the flat (no distortion image)
    tel-atm
    tel*wfs
    cube_ref = wfs.cam.cube
    cube_ref = np.delete(cube_ref, removed_frames, axis=0)
    frame_ref = wfs.cam.frame


    def process(b):

        atmLocal = copy.deepcopy(atm)
        telLocal = copy.deepcopy(tel)
        wfsLocal = copy.deepcopy(wfs)
        # Get phase mask on atmosphere
        atmLocal.OPD_no_pupil = bases[b, :, :] * atmLocal.wavelength / 2 / np.pi
        atmLocal.OPD = atmLocal.OPD_no_pupil * telLocal.pupil
        # Apply atmosphere to telescope
        telLocal + atmLocal
        # Propagate to wfs
        telLocal * wfsLocal
        # Grab image
        cube = wfsLocal.cam.cube
        cube = np.delete(cube, removed_frames, axis=0)

        flat_ref = np.sum(cube_ref, axis=0)
        flat = np.sum(cube, axis=0)
        t = np.sum(flat)
        t_ref = np.sum(flat_ref)
        I = (flat / np.sum(flat)) - (flat_ref / np.sum(flat_ref))
        I_flat = I.flatten()
        I_flat_normed = I_flat / np.linalg.norm(I_flat)
        #D[:, b] = I_flat

        # Remove distortion
        telLocal - atmLocal

        return I_flat

    D = Parallel(n_jobs=4)(delayed(process)(i) for i in range(bases.shape[0]))


    return np.array(D)



atm = temp_atm(tel)


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


removed_frames=[] #[5,6,7,17,18,19,29,30,31,41,42,43]

#D = generateInteractionMatrix(atm, tel, wfs, bases, removed_frames=removed_frames)
D = PWFS_CMOS_NOISE_PROOF_may2022_tools.generateInteractionMatrix(atm,tel,wfs,bases,removed_frames=removed_frames, amplitude=(30*1e-9))

def calcBDiag(atm, tel, wfs, bases, removed_frames=[]):
    #D = generateInteractionMatrix(atm, tel, wfs, bases, removed_frames=removed_frames)
    D = PWFS_CMOS_NOISE_PROOF_may2022_tools.generateInteractionMatrix(atm,tel,wfs,bases,removed_frames=removed_frames, amplitude=(30*1e-9))
    D_plus = inv(D.T @ D) @ D.T
    B = D_plus @ D_plus.T

    return np.diag(B)

#D = D/np.linalg.norm(D)

D_plus = inv(D.T@D)@D.T

# # Test to see if it worked:
# atm.OPD_no_pupil = fourierBases[0,:,:] * atm.wavelength / 2 / np.pi
# atm.OPD = atm.OPD_no_pupil * tel.pupil
# tel-atm
# tel*wfs
#
# frame_ref = np.sum(np.delete(wfs.cam.cube, removed_frames, axis=0), axis=0)
#
# tel+atm
# tel*wfs
#
# frame = np.sum(np.delete(wfs.cam.cube, removed_frames, axis=0), axis=0)
#
# P_p = ( frame/ np.sum(frame)) - frame_ref/np.sum(frame_ref)
# P_p = P_p.flatten()
# P = np.zeros((P_p.shape[0],1))
# P[:,0] = P_p
#
# tel-atm
#
# a = D_plus@P
# #Done Test


B = D_plus@D_plus.T


B_diag = np.diag(B)

# Remove 2 frames per face
remove_frames = [6,18,30,42]
# Recalc D
#D_m2 = generateInteractionMatrix(atm, tel, wfs, bases, removed_frames=remove_frames)
D_m2 = PWFS_CMOS_NOISE_PROOF_may2022_tools.generateInteractionMatrix(atm,tel,wfs,bases,removed_frames=removed_frames, amplitude=(30*1e-9))
# Recalc B
D_plus_m2 = inv(D_m2.T@D_m2)@D_m2.T
B_m2 = D_plus_m2@D_plus_m2.T
B_diag_m2 = np.diag(B_m2)


# Remove 4 frames per face
remove_frames = [5,6,7,17,18,19,29,30,31,41,42,43]
# Recalc D
#D_m4 = generateInteractionMatrix(atm, tel, wfs, bases, removed_frames=remove_frames)
D_m4 = PWFS_CMOS_NOISE_PROOF_may2022_tools.generateInteractionMatrix(atm,tel,wfs,bases,removed_frames=removed_frames, amplitude=(30*1e-9))
# Recalc B
D_plus_m4 = inv(D_m4.T@D_m4)@D_m4.T
B_m4 = D_plus_m4@D_plus_m4.T
B_diag_m4 = np.diag(B_m4)

# Remove 6 frames per face
remove_frames = [4,5,6,7,8,16,17,18,19,20,28,29,30,31,32,40,41,42,43,44]
# Recalc D
#D_m6 = generateInteractionMatrix(atm, tel, wfs, bases, removed_frames=remove_frames)
D_m6 = PWFS_CMOS_NOISE_PROOF_may2022_tools.generateInteractionMatrix(atm,tel,wfs,bases,removed_frames=removed_frames, amplitude=(30*1e-9))
# Recalc B
D_plus_m6 = inv(D_m6.T@D_m6)@D_m6.T
B_m6 = D_plus_m6@D_plus_m6.T
B_diag_m6 = np.diag(B_m6)

# Remove 8 frames per face
remove_frames = [3,4,5,6,7,8,9,15,16,17,18,19,20,21,27,28,29,30,31,32,33,39,40,41,42,43,44,45]
#remove_frames = list(range(48))
# Recalc D
#D_m8 = generateInteractionMatrix(atm, tel, wfs, bases, removed_frames=remove_frames)
D_m8 = PWFS_CMOS_NOISE_PROOF_may2022_tools.generateInteractionMatrix(atm,tel,wfs,bases,removed_frames=removed_frames, amplitude=(30*1e-9))
# Recalc B
D_plus_m8 = inv(D_m8.T@D_m8)@D_m8.T
B_m8 = D_plus_m8@D_plus_m8.T
B_diag_m8 = np.diag(B_m8)

# Remove 9 frames per face
remove_frames = [2,3,4,5,6,7,8,9,10,14,15,16,17,18,19,20,21,22,26,27,28,29,30,31,32,33,34,38,39,40,41,42,43,44,45,46]
#remove_frames = list(range(48))
# Recalc D
#D_m9 = generateInteractionMatrix(atm, tel, wfs, bases, removed_frames=remove_frames)
D_m9 = PWFS_CMOS_NOISE_PROOF_may2022_tools.generateInteractionMatrix(atm,tel,wfs,bases,removed_frames=removed_frames, amplitude=(30*1e-9))
# Recalc B
D_plus_m9 = inv(D_m9.T@D_m9)@D_m9.T
B_m9 = D_plus_m9@D_plus_m9.T
B_diag_m9 = np.diag(B_m9)

# Remove 11 frames per face
remove_frames = [1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19,20,21,22,23,25,26,27,28,29,30,31,32,33,34,35,37,38,39,40,41,42,43,44,45,46,47]
#remove_frames = list(range(48))
# Recalc D
#D_m11 = generateInteractionMatrix(atm, tel, wfs, bases, removed_frames=remove_frames)
D_m11 = PWFS_CMOS_NOISE_PROOF_may2022_tools.generateInteractionMatrix(atm,tel,wfs,bases,removed_frames=removed_frames, amplitude=(30*1e-9))
# Recalc B
D_plus_m11 = inv(D_m11.T@D_m11)@D_m11.T
B_m11 = D_plus_m11@D_plus_m11.T
B_diag_m11 = np.diag(B_m11)

font = {'family': 'normal',
        'weight': 'bold',
        'size': 15}

mpl.rc('font', **font)

fig = plt.figure(4, figsize=(17, 10))
ax = plt.subplot(1,3,1)
ax.bar(np.arange(len(B_diag)), B_diag, label="No frames removed", color='b')
ax.bar(np.arange(len(B_diag_m2)), B_diag_m2, label="1/face removed", color='tab:orange')
ax.bar(np.arange(len(B_diag_m4)), B_diag_m4, label="3/face removed", color='g')
ax.bar(np.arange(len(B_diag_m6)), B_diag_m6, label="5/face removed", color='r')
ax.bar(np.arange(len(B_diag_m8)), B_diag_m8, label="7/face removed", color='tab:purple')
ax.bar(np.arange(len(B_diag_m9)), B_diag_m9, label="9/face removed", color='tab:brown')
ax.bar(np.arange(len(B_diag_m11)), B_diag_m11, label="11/face removed", color='tab:pink')
ax.xaxis.set_major_locator(MaxNLocator(integer=True))

# for m in range(len(B_diag)):
#     per = (B_diag_m2[m] / B_diag[m]) * 100
#     ax.text(m, B_diag_m2[m], f'{per:.2f} %', ha='center', va='center')
#
# for m in range(len(B_diag)):
#     per = (B_diag_m4[m] / B_diag[m]) * 100
#     ax.text(m, B_diag_m4[m], f'{per:.2f} %', ha='center', va='center')

ax.set_xlabel("Mode")
ax.set_ylabel("Noise factor (unitless?)")
ax.set_title("Noise factor as a function of the mode")
ax.legend()


ax2 = plt.subplot(1,3,2)

ax2.plot((B_diag_m2 / B_diag) * 100, label="1/face removed", color='tab:orange')
ax2.plot((B_diag_m4 / B_diag) * 100, label="3/face removed", color='g')
ax2.plot((B_diag_m6 / B_diag) * 100, label="5/face removed", color='r')
ax2.plot((B_diag_m8 / B_diag) * 100, label="7/face removed", color='tab:purple')
ax2.plot((B_diag_m9 / B_diag) * 100, label="9/face removed", color='tab:brown')
ax2.plot((B_diag_m11 / B_diag) * 100, label="11/face removed", color='tab:pink')

ax2.axhline(100, lw=3, linestyle='--', color='k')

ax2.set_xlabel("Mode number")
ax2.set_ylabel("Noise factor as % of no face removed")
ax2.set_title('Percentage of the noise factor relative to \n no frames removed as a function of the mode')

ax2.legend()
ax2.xaxis.set_major_locator(MaxNLocator(integer=True))



# fig = plt.figure(5, figsize=(17, 10))
# ax = plt.subplot(1,3,1)
#
# B_diag_adj = B_diag
# ax.bar(np.arange(len(B_diag)), B_diag, label="No frames removed", color='b')
# bar1 = ax.bar(np.arange(len(B_diag_adj)), B_diag_adj, label="# removed", color='tab:orange')
#
# ax2 = plt.subplot(1,3,2)
# ax2.axhline(100, lw=3, linestyle='--', color='k')
# plot1 = ax2.plot((B_diag_adj / B_diag) * 100, label="1/face removed", color='tab:orange')
#
ax3 = plt.subplot(1,3,3, projection='polar')

remove_frames = [5,6,7,17,18,19,29,30,31,41,42,43]
selected = np.delete(wfs.thetaModulation,remove_frames)
removed = wfs.thetaModulation[remove_frames]
colorPie = ['b']*len(wfs.thetaModulation)
frame_selector = ax3.bar(wfs.thetaModulation, 1, width=(2*np.pi/wfs.thetaModulation.shape[0]), color=colorPie, edgecolor='k', picker=1)
ax3.bar(removed, 1, width=(2*np.pi/wfs.thetaModulation.shape[0]), color="k", edgecolor='r')
#
# def on_pick(event):
#     line = event.artist
#     xdata = line.get_x()
#     dif = wfs.thetaModulation[1]-wfs.thetaModulation[0]
#     xdata = xdata+(dif/2)
#     index = np.abs(wfs.thetaModulation-xdata).argmin()
#
#     if colorPie[index] == 'b':
#         colorPie[index] = 'k'
#     else:
#         colorPie[index] = 'b'
#
#
#     frame_selector[index].set_color(colorPie[index])
#     removed_frames = np.argwhere(np.array(colorPie) == 'k')
#
#     B_diag_adj = calcBDiag(atm, tel, wfs, bases, removed_frames=removed_frames)
#     for i in range(len(B_diag_adj)):
#         bar1[i].set_height(B_diag_adj[i])
#     data = (B_diag_adj / B_diag) * 100
#     plot1[0].set_ydata(data)
#     ax2.set_ylim([np.min(data), np.max([np.max(data), 110])])
#
#
# cid = fig.canvas.mpl_connect('pick_event', on_pick)

# np.savetxt("D_no_remove_KL.csv", D, delimiter=",")
# np.savetxt("D_1_remove_KL.csv", D_m2, delimiter=",")
# np.savetxt("D_3_remove_KL.csv", D_m4, delimiter=",")
# np.savetxt("D_5_remove_KL.csv", D_m6, delimiter=",")
# np.savetxt("D_7_remove_KL.csv", D_m8, delimiter=",")
# np.savetxt("D_9_remove_KL.csv", D_m9, delimiter=",")
# np.savetxt("D_11_remove_KL.csv", D_m11, delimiter=",")

plt.show(block=True)



'''
#%%
tel*wfs

#tel+atm
tel*wfs
ax5 = plt.subplot(3,2,5)

ax5.imshow(wfs.cam.frame)
ax5.set_title('WFS Camera Frame')


cube_ref = wfs.cam.cube


tel+atm
tel*wfs
cube = wfs.cam.cube

image_cube_o = np.abs(wfs.modulation_camera_frame)
sorted_idx = np.argsort( np.array(wfs.modulation_camera_frame_phase))
mod_camera_frame_phase = np.array(wfs.modulation_camera_frame_phase)
image_cube = np.zeros(image_cube_o.shape)
for i in range(len(wfs.thetaModulation)) :
    val_phase = (2*np.pi-wfs.thetaModulation[i])
    idx = np.where(mod_camera_frame_phase==val_phase)
    image_cube[-i,:,:] = image_cube_o[idx,:,:]

#image_cube = image_cube[sorted_idx,:,:]


import matplotlib.cm as cm
import matplotlib.animation as animation

frames = [] # for storing the generated images
beam = []
distance  = (atm.wavelength / param['diameter'] ) * param['modulation']

# fig = plt.figure(10)
# fig_3, ax_i = plt.subplots(5,5)
# for i in range(25):
#     ax_i[i%5,i//5].imshow(phaseMask_o[i,:,:])

fig = plt.figure(2)
anim_running = True

ax1 = plt.subplot(2, 3, 1)
ax1.set_title("Abberated pyramid view")
ax1_d = plt.subplot(2, 3, 2)
ax1_d.set_title("Difference pyramid view")
ax1_r = plt.subplot(2, 3, 3)
ax1_r.set_title("Flat pyramid view")
ax2 = plt.subplot(2, 3, 4, projection='polar')
ax2.set_title(r'Beam position and $\Delta \mathrm{I} ( \phi )$')
ax3 = plt.subplot(2, 3, 5)
ax3.set_title(r'$\Delta \mathrm{I} ( \phi )$ as a function of the frame number (max = ' + str(cube.shape[0]) + ')')
ax4 = plt.subplot(2, 3, 6)
ax4.set_title("PSF position relative to the pyramid peak")

# set the spacing between subplots
plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.4,
                    hspace=0.4)

delta_I = np.zeros((cube.shape[0]))
cube_diff = np.zeros(cube.shape)
for i in range(cube.shape[0]):
    delta_I[i] = np.sqrt(np.sum(((cube[i,:,:]/np.sum(cube[i,:,:])) - (cube_ref[i,:,:]/np.sum(cube_ref[i,:,:])))**2))
    cube_diff[i,:,:] = cube[i,:,:]-cube_ref[i,:,:]

ax3.bar(list(range(cube.shape[0])), delta_I)
ax2.bar(2 * np.pi - wfs.thetaModulation, delta_I, width=(2*np.pi/cube.shape[0]), edgecolor="c")

im1 = ax1.imshow(cube[0, :, :], cmap=cm.Greys_r)
im2 = ax1_r.imshow(cube_ref[0, :, :], cmap=cm.Greys_r)
im3 = ax1_d.imshow(cube_diff[0, :, :], cmap=cm.Greys_r)
im4 = ax2.scatter((2 * np.pi - wfs.thetaModulation[0]), distance, s=200)
im5 = ax3.axvline(0, color='r')
ax2.axvline(0,lw=2, alpha=0.5, color='k')
ax2.axvline(np.pi*0.5,lw=2, alpha=0.5, color='k')
ax2.axvline(np.pi,lw=2, alpha=0.5, color='k')
ax2.axvline(np.pi*1.5,lw=2, alpha=0.5, color='k')
im6 = ax2.axvline(0, lw=3, color='r')
im7 = ax4.imshow(image_cube[0, :, :], cmap=cm.Greys_r)
mid = wfs.nRes/2.0
ax4.axvline(mid,lw=2, alpha=0.5, color='r')
ax4.axhline(mid,lw=2, alpha=0.5, color='r')


im1_cbar = plt.colorbar(im1, ax=ax1)
im1_cbar.ax.set_autoscale_on(True)
im2_cbar = plt.colorbar(im2, ax=ax1_r)
im2_cbar.ax.set_autoscale_on(True)
im3_cbar = plt.colorbar(im3, ax=ax1_d)
im3_cbar.ax.set_autoscale_on(True)

# for i in range(cube.shape[0]):
#     im1 = ax1.imshow(cube[i, :, :], cmap=cm.Greys_r)
#     im1.set_clim(vmin=cube[i, :, :].min(), vmax=cube[i, :, :].max())
#     im2 = ax1_r.imshow(cube_ref[i, :, :], cmap=cm.Greys_r)
#     im2.set_clim(vmin=cube_ref[i, :, :].min(), vmax=cube_ref[i, :, :].max())
#     im3 = ax1_d.imshow(cube_diff[i, :, :], cmap=cm.Greys_r)
#     im3.set_clim(vmin=cube_diff[i, :, :].min(), vmax=cube_diff[i, :, :].max())
#     im4 = ax2.scatter((2*np.pi - wfs.thetaModulation[i]), distance, s=200)
#     im5 = ax3.axvline(i)
#
#     frames.append([im1, im2, im3, im4, im5])

def funcAnim(i):
    #im1 = ax1.imshow(cube[i, :, :], cmap=cm.Greys_r)
    im1.set_data(cube[i, :, :])
    im1.set_clim(vmin=cube[i, :, :].min(), vmax=cube[i, :, :].max())
    #im1_cbar.set_clim(vmin=cube[i, :, :].min(), vmax=cube[i, :, :].max())
    #im2 = ax1_r.imshow(cube_ref[i, :, :], cmap=cm.Greys_r)
    im2.set_data(cube_ref[i, :, :])
    im2.set_clim(vmin=cube_ref[i, :, :].min(), vmax=cube_ref[i, :, :].max())
    #im2_cbar.set_clim(vmin=cube_ref[i, :, :].min(), vmax=cube_ref[i, :, :].max())
    #im3 = ax1_d.imshow(cube_diff[i, :, :], cmap=cm.Greys_r)
    im3.set_data(cube_diff[i, :, :])
    im3.set_clim(vmin=cube_diff[i, :, :].min(), vmax=cube_diff[i, :, :].max())
    #im3_cbar.set_clim(vmin=cube_diff[i, :, :].min(), vmax=cube_diff[i, :, :].max())
    #im7 = ax4.imshow(image_cube[i, :, :], cmap=cm.Greys_r)
    im7.set_data(image_cube[i, :, :])
    im7.set_clim(vmin=image_cube[i, :, :].min(), vmax=image_cube[i, :, :].max())

    ax4.set_xlim(mid - 20, mid + 20)
    ax4.set_ylim(mid - 20, mid + 20)
    im4.set_offsets([(2 * np.pi - wfs.thetaModulation[i]), distance])
    im5.set_xdata(list(range(len(wfs.thetaModulation)))[i])
    im6.set_xdata((2 * np.pi - wfs.thetaModulation[i]))
    return [im1, im2, im3, im4, im5, im6, im7]

def update_time():
    t = -1
    t_max = int(cube.shape[0])-1
    while t<t_max:
        t += anim.direction
        yield t

#anim = animation.ArtistAnimation(fig, frames, interval=700, blit=False, repeat_delay=700)
anim = animation.FuncAnimation(fig, funcAnim, frames=update_time,interval=500, blit=False, repeat_delay=500)
anim.direction = 1

# writervideo = animation.FFMpegWriter(fps=30)
# anim.save("pyr_anim.mp4", writer=writervideo)

def onClick(event):
    global anim_running
    if anim_running:
        anim.event_source.stop()
        anim_running = False
    else:
        anim.event_source.start()
        anim_running = True

cid= fig.canvas.mpl_connect('button_press_event', onClick)

def on_press(event):
    if event.key.isspace():
        if anim.running:
            anim.event_source.stop()
        else:
            anim.event_source.start()
        anim.running ^= True
    elif event.key == 'left':
        anim.direction = -1
    elif event.key == 'right':
        anim.direction = +1

    # Manually update the plot
    if event.key in ['left','right']:
        t = anim.frame_seq.__next__()
        funcAnim(t)
        plt.draw()

fig.canvas.mpl_connect('key_press_event', on_press)

plt.pause(1) # <---- add pause

plt.show(block=True)

'''

'''

tel-atm
#%% ZERNIKE Polynomials
from AO_modules.Zernike import Zernike
# create Zernike Object
Z = Zernike(tel,300)
# compute polynomials for given telescope
Z.computeZernike(tel)

# mode to command matrix to project Zernike Polynomials on DM
M2C_zernike = np.linalg.pinv(np.squeeze(dm.modes[tel.pupilLogical,:]))@Z.modes

# show the first 10 zernikes
dm.coefs = M2C_zernike[:,:10]
tel*dm
# displayMap(tel.OPD)

#%% to manually measure the interaction matrix

# amplitude of the modes in m
stroke=1e-9
# Modal Interaction Matrix
from AO_modules.calibration.InteractionMatrix import interactionMatrix


#%%
M2C_zonal = np.eye(dm.nValidAct)
# zonal interaction matrix
calib_zonal = interactionMatrix(  ngs            = ngs,\
                            atm            = atm,\
                            tel            = tel,\
                            dm             = dm,\
                            wfs            = wfs,\
                            M2C            = M2C_zonal,\
                            stroke         = stroke,\
                            nMeasurements  = 25,\
                            noise          = 'off')

# plt.figure(9)
# plt.plot(np.std(calib_zonal.D,axis=0))
# plt.xlabel('Mode Number')
# plt.ylabel('WFS slopes STD')

#%%
# Modal interaction matrix
calib_zernike = interactionMatrix(  ngs            = ngs,\
                            atm            = atm,\
                            tel            = tel,\
                            dm             = dm,\
                            wfs            = wfs,\
                            M2C            = M2C_zernike,\
                            stroke         = stroke,\
                            nMeasurements  = 50,\
                            noise          = 'off')

# plt.figure(10)
# plt.plot(np.std(calib_zernike.D,axis=0))
# plt.xlabel('Mode Number')
# plt.ylabel('WFS slopes STD')
#
# plt.show(block=True)
#%%
# These are the calibration data used to close the loop
calib_CL    = calib_zernike
M2C_CL      = M2C_zernike

plt.close('all')

# combine telescope with atmosphere
tel+atm

# initialize DM commands
dm.coefs=0
ngs*tel*dm*wfs

plt.ion()
# setup the display
fig = plt.figure(79)
ax1 = plt.subplot(3, 3, 1)
im_atm = ax1.imshow(tel.src.phase)
plt.colorbar(im_atm)
plt.title('Turbulence phase [rad]')

ax2 = plt.subplot(3, 3, 2)
im_dm = ax2.imshow(dm.OPD * tel.pupil)
plt.colorbar(im_dm)
plt.title('DM phase [rad]')
tel.computePSF(zeroPaddingFactor=6)

ax4 = plt.subplot(3, 3, 3)
im_PSF_OL = ax4.imshow(tel.PSF_trunc)
plt.colorbar(im_PSF_OL)
plt.title('OL PSF')

ax3 = plt.subplot(3, 3, 5)
im_residual = ax3.imshow(tel.src.phase)
plt.colorbar(im_residual)
plt.title('Residual phase [rad]')

ax5 = plt.subplot(3, 3, 4)
im_wfs_CL = ax5.imshow(wfs.cam.frame)
plt.colorbar(im_wfs_CL)
plt.title('Pyramid Frame CL')

ax6 = plt.subplot(3, 3, 6)
im_PSF = ax6.imshow(tel.PSF_trunc)
plt.colorbar(im_PSF)
plt.title('CL PSF')

# ax7 = plt.subplot(3, 3, 7)
# pl_spa_freq = ax7.semilogy(np.arange(0, 2 * np.pi, 0.01), spectrum(np.arange(0, 2 * np.pi, 0.01), atm))
# # plt.plot(pl_spa_freq)
# plt.title('Spatial frequency of the residual phase')

plt.show()

param['nLoop'] = 100
# allocate memory to save data
SR = np.zeros(param['nLoop'])
total = np.zeros(param['nLoop'])
residual = np.zeros(param['nLoop'])
wfsSignal = np.arange(0, wfs.nSignal) * 0

# loop parameters
gainCL = 0.6
wfs.cam.photonNoise = True
display = True

reconstructor = M2C_CL @ calib_CL.M

filenames = []

for i in range(param['nLoop']):
    a = time.time()
    # update phase screens => overwrite tel.OPD and consequently tel.src.phase
    atm.update()
    # save phase variance
    total[i] = np.std(tel.OPD[np.where(tel.pupil > 0)]) * 1e9
    # save turbulent phase
    turbPhase = tel.src.phase
    if display == True:
        # compute the OL PSF and update the display
        tel.computePSF(zeroPaddingFactor=6)
        im_PSF_OL.set_data(np.log(tel.PSF_trunc / tel.PSF_trunc.max()))
        im_PSF_OL.set_clim(vmin=-3, vmax=0)

    # propagate to the WFS with the CL commands applied
    tel * dm * wfs

    # save the DM OPD shape
    dmOPD = tel.pupil * dm.OPD * 2 * np.pi / ngs.wavelength

    t_sim = np.matmul(reconstructor, wfsSignal)
    g_sim = reconstructor @ wfsSignal

    dm.coefs = dm.coefs - gainCL * np.matmul(reconstructor, wfsSignal)
    # store the slopes after computing the commands => 2 frames delay
    wfsSignal = wfs.signal
    b = time.time()
    print('Elapsed time: ' + str(b - a) + ' s')
    # update displays if required
    if display == True:
        # Turbulence
        im_atm.set_data(turbPhase)
        im_atm.set_clim(vmin=turbPhase.min(), vmax=turbPhase.max())
        # WFS frame
        C = wfs.cam.frame
        im_wfs_CL.set_data(C)
        im_wfs_CL.set_clim(vmin=C.min(), vmax=C.max())
        # DM OPD
        im_dm.set_data(dmOPD)
        im_dm.set_clim(vmin=dmOPD.min(), vmax=dmOPD.max())

        # residual phase
        D = tel.src.phase
        D = D - np.mean(D[tel.pupil])
        im_residual.set_data(D)
        im_residual.set_clim(vmin=D.min(), vmax=D.max())

        # # Take the fourier transform of the image.
        # F1 = fftpack.fft2(D)
        # # Now shift the quadrants around so that low spatial frequencies are in
        # # the center of the 2D fourier transformed image.
        # F2 = fftpack.fftshift(F1)
        # # Calculate a 2D power spectrum
        # psd2D = np.abs(F2) ** 2
        # # Calculate the azimuthally averaged 1D power spectrum
        # psd1D = radialProfile.azimuthalAverage(psd2D)
        # pl_spa_freq[0].set_data(np.arange(0, 2 * np.pi, 0.01), spectrum(np.arange(0, 2 * np.pi, 0.01), atm))
        # #pl_spa_freq.set_clim(vmin=spa_freqx.min(), vmax=spa_freqx.max())

        tel.computePSF(zeroPaddingFactor=6)
        im_PSF.set_data(np.log(tel.PSF_trunc / tel.PSF_trunc.max()))
        im_PSF.set_clim(vmin=-4, vmax=0)
        plt.draw()
        plt.show()
        plt.pause(0.001)

    # create file name and append it to a list
    filename = f'{i}.png'
    filenames.append(filename)

    # save frame
    plt.savefig(filename)
    # plt.close()

    SR[i] = np.exp(-np.var(tel.src.phase[np.where(tel.pupil == 1)]))
    residual[i] = np.std(tel.OPD[np.where(tel.pupil > 0)]) * 1e9
    OPD = tel.OPD[np.where(tel.pupil > 0)]

    print('Loop' + str(i) + '/' + str(param['nLoop']) + ' Turbulence: ' + str(total[i]) + ' -- Residual:' + str(
        residual[i]) + '\n')

# build gif
with imageio.get_writer('mygif.gif', mode='I') as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)

# Remove files
for filename in set(filenames):
    os.remove(filename)

# %%
plt.figure()
plt.plot(total)
plt.plot(residual)
plt.xlabel('Loop')
plt.ylabel('WFE [nm]')

plt.show(block=True)

'''