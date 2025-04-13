import copy
from functools import lru_cache
import aotools.functions.zernike
from joblib import Parallel, delayed
import imageio, os
from skimage.util import random_noise
import contextlib, io, sys
from tqdm import tqdm

# commom modules
#import matplotlib.backends.backend_qt5agg
from matplotlib.ticker import MaxNLocator
import matplotlib as mpl
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}

mpl.rc('font', **font)
import matplotlib.pyplot as plt
#mpl.use('Qt5Agg')
import numpy             as np
from numpy.linalg import inv
import time
plt.ion()
#import __load__psim
#__load__psim.load_psim()

from OOPAO.Atmosphere       import Atmosphere
from OOPAO.TRM_Pyramid import TRM_Pyramid
from OOPAO.DeformableMirror import DeformableMirror
from OOPAO.MisRegistration  import MisRegistration
from OOPAO.Telescope        import Telescope
from OOPAO.Source           import Source
from OOPAO.Pyramid import  Pyramid


class dummy_atm():
    """
    Dummy atmosphere class to more easily use the custom bases we want as turbulence
    """
    def __init__(self, tel):
        self.OPD = None
        self.OPD_no_pupil = None
        self.wavelength = 790*1e-9
        self.tel = tel
        self.r0 = 0.186
        self.L0 = 30
        self.tag = 'atmosphere'

    def __mul__(self,obj):
        obj.OPD=self.OPD
        obj.OPD_no_pupil=self.OPD_no_pupil
        obj.isPaired=True
        return obj

    def update(self):
        if self.tel.isPaired:
            self * self.tel



def fourierPhaseMask(wavelengthPix, angleRad, size=120):
    """
    Generates the Fourier Phase masks
    Parameters
    ----------
    wavelengthPix : Number of pixels for the wavelength, if equal to size, then its first harmonic
    angleRad : angle in radian of the spatial frequency
    size : resolution of the image in pixels

    Returns
    -------
    2D array (size,size) of the phase mask
    """
    x = np.arange(-(size/2), (size/2), 1)
    X, Y = np.meshgrid(x, x)

    if wavelengthPix is np.NaN:
        phaseMask = (X+Y)*0
        return phaseMask
    else:
        phaseMask = np.sin(2*np.pi*(X*np.cos(angleRad) + Y*np.sin(angleRad)) / wavelengthPix)
        return phaseMask * np.sqrt(2) / size

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
    maxCycles = 20
    cycles = np.linspace(0.001, maxCycles, bases)
    pixel_res_cyc = resolution / cycles
    factor = np.sqrt(2) # To have an RMS = 1
    fourierBases = np.zeros((bases*2,resolution,resolution))
    for i in range(bases):
        j = i*2

        fourierBases[  j, :, :] = fourierPhaseMask(pixel_res_cyc[i],       0, size=resolution) * factor  # Horizontal
        fourierBases[j+1, :, :] = fourierPhaseMask(pixel_res_cyc[i], np.pi/4, size=resolution) * factor  # Vertical

    return fourierBases


def generateFourierBasesWithAngle(bases, resolution, angle, maxCycles):
    cycles = np.linspace(0.001, maxCycles, bases)
    pixel_res_cyc = resolution / cycles
    factor = np.sqrt(2) # To have an RMS = 1
    fourierBases = np.zeros((bases, resolution, resolution))
    for i in range(bases):
        fourierBases[  i, :, :] = fourierPhaseMask(pixel_res_cyc[i],  angle, size=resolution) * factor

    return fourierBases, cycles

def generateBases(num, res, baseType, display=True, scale=False):
    """
    Generate the bases to use for the interaction matrix
    Parameters
    ----------
    num : Number of bases
    res : resolution of the bases
    baseType : How to generate the bases. 2 choices: Fourier or KL
    display : if True, displays the bases when done. blocking

    Returns
    -------
    The bases in an (num, res^2) format
    """
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

    if scale:
        max_ax0 = np.amax(np.abs(bases), axis=(1,2))
        bases = bases / max_ax0[:, np.newaxis, np.newaxis]
    return bases

def displayBases(bases):
    """
    Display the bases with matplotlib. Blocking
    Parameters
    ----------
    bases : The bases to display

    Returns
    -------
    None

    """
    numBases = bases.shape[0]
    side = int(np.ceil(np.sqrt(numBases)))
    fig, ax_i = plt.subplots(side, side, figsize=(10, 7))


    maxValue = np.max(bases)
    minValue = np.min(bases)
    avgValue = np.mean(np.mean(bases,axis=0))
    t = np.sum(bases**2, axis=(1,2))/(bases.shape[1]*bases.shape[2])
    RMSValue = np.mean(np.sqrt(t))
    fig.suptitle(f"Bases used for the interaction matrix\nMax: {maxValue:.2f} nm, Min: {minValue:.2f} nm, Avg: {avgValue:.2f} nm, RMS: {RMSValue:.2f}", fontsize=18)
    for x in range(side):
        for y in range(side):
            i = (x * side) + y
            ax_i[x, y].axis('off')
            if i < numBases:
                ax_i[x, y].imshow(bases[i, :, :])
            else:
                ax_i[x, y].set_visible(False)


def generateInteractionMatrix(atm, tel, wfs, bases, removed_frames=None, amplitude=(30*1e-9)):
    """
    Generate the interaction matrix for the system. uses the bases provided
    Parameters
    ----------
    atm : atmosphere (refer to temp_atm or AO_modules.Atmosphere)
    tel : Telescope object (refer to AO_modules.Telescope)
    wfs : Wavefront sensor object (refer to AO_modules.Pyramid)
    bases : Bases to be used for the system. (num_of_bases, res, res)
    removed_frames : list of the indices of the frames to be removed

    Returns
    -------
    The interaction matrix (res^2, num_of_bases)
    """

    if removed_frames is None:
        removed_frames = []

    # if isinstance(removed_frames[0], list):
    #     D = np.zeros((len(removed_frames), wfs.cam.resolution * wfs.cam.resolution, bases.shape[0]))
    # else:
    #D = np.zeros((wfs.cam.resolution * wfs.cam.resolution, bases.shape[0]))
    D = np.zeros((np.sum(wfs.validSignal), bases.shape[0]))
    # Start by making the flat (no distortion image)
    tel-atm
    tel*wfs
    cube_ref_no_remove = wfs.cam.cube

    # Iterate through the bases
    for b in range(bases.shape[0]):
        # Get phase mask on atmosphere
        atm.OPD_no_pupil = bases[b,:,:] * amplitude
        atm.OPD = atm.OPD_no_pupil * tel.pupil
        # Apply atmosphere to telescope
        tel+atm
        # Propagate to wfs
        tel*wfs
        # Grab image
        cube_no_remove_p = wfs.cam.cube

        tel - atm

        atm.OPD_no_pupil = bases[b,:,:] * -amplitude
        atm.OPD = atm.OPD_no_pupil * tel.pupil
        # Apply atmosphere to telescope
        tel+atm
        # Propagate to wfs
        tel*wfs
        # Grab image
        cube_no_remove_n = wfs.cam.cube

        # if isinstance(removed_frames[0], list):
        #     for i in range(len(removed_frames)):
        #         ref = np.delete(cube_ref_no_remove, removed_frames[i], axis=0)
        #         img = np.delete(cube_no_remove, removed_frames[i], axis=0)
        #
        #         flat_ref = np.sum(ref, axis=0)
        #         flat = np.sum(img, axis=0)
        #
        #         I = (flat / np.sum(flat)) - (flat_ref / np.sum(flat_ref))
        #         I_flat = I.flatten()
        #
        #         D[i, :, b] = I_flat / amplitude
        # else:
        ref = np.delete(cube_ref_no_remove, removed_frames, axis=0)
        img_p = np.delete(cube_no_remove_p, removed_frames, axis=0)
        img_n = np.delete(cube_no_remove_n, removed_frames, axis=0)

        flat_ref = np.sum(ref, axis=0)
        flat_ref = flat_ref[np.where(wfs.validSignal == 1)]

        flat_p = np.sum(img_p, axis=0)
        flat_p = flat_p[np.where(wfs.validSignal == 1)]

        flat_n = np.sum(img_n, axis=0)
        flat_n = flat_n[np.where(wfs.validSignal == 1)]

        I_p = (flat_p / np.sum(flat_p)) - (flat_ref / np.sum(flat_ref))
        I_flat_p = I_p.flatten()

        I_n = (flat_n / np.sum(flat_n)) - (flat_ref / np.sum(flat_ref))
        I_flat_n = I_n.flatten()

        D[:, b] = 0.5 * (I_flat_p-I_flat_n) / amplitude

        # D[:, b] = I_flat / amplitude

        #Remove distortion
        tel - atm
    return D


def calcWeightsPerFramePerBase(param, bases, D_amp=(1e-9)):
    # create the Telescope object
    tel = Telescope(resolution=param['resolution'],
                    diameter=param['diameter'],
                    samplingTime=param['samplingTime'],
                    centralObstruction=param['centralObstruction'])

    # %% -----------------------     NGS   ----------------------------------
    # create the Source object
    ngs = Source(optBand=param['opticalBand'],
                 magnitude=param['magnitude'])

    # combine the NGS to the telescope using '*' operator:
    ngs * tel

    tel.computePSF(zeroPaddingFactor=6)

    # %% -----------------------     ATMOSPHERE   ---------------------------------

    "Dummy atmosphere to be able to fit whatever aberration we want"
    atm = dummy_atm(tel)

    wfs = TRM_Pyramid(nSubap=param['nSubaperture'],
                      telescope=tel,
                      modulation=param['modulation'],
                      lightRatio=param['lightThreshold'],
                      pupilSeparationRatio=param['pupilSeparationRatio'],
                      calibModulation=param['calibrationModulation'],
                      psfCentering=param['psfCentering'],
                      edgePixel=param['edgePixel'],
                      extraModulationFactor=param['extraModulationFactor'],
                      postProcessing=param['postProcessing'],
                      nTheta_user_defined=param["nTheta_user_defined"],
                      temporal_weights_settings=None)

    W = np.zeros((bases.shape[0], wfs.nTheta))
    # D = np.zeros((np.sum(wfs.validSignal), bases.shape[0]))
    # Start by making the flat (no distortion image)
    tel - atm
    tel * wfs
    cube_ref_no_remove = wfs.cam.cube

    # Iterate through the bases
    for b in range(bases.shape[0]):
        # Get phase mask on atmosphere
        atm.OPD_no_pupil = bases[b, :, :] * D_amp
        atm.OPD = atm.OPD_no_pupil * tel.pupil
        # Apply atmosphere to telescope
        tel + atm
        # Propagate to wfs
        tel * wfs
        # Grab image
        cube_no_remove = wfs.cam.cube

        weights = calcDeltaIPerFrame(cube=cube_no_remove, cube_ref=cube_ref_no_remove, amp=D_amp)
        weights = weights / np.max(weights)

        W[b, :] = weights

        # Remove distortion
        tel - atm

    return W


def calcWeightsPerFramePerBaseWithDM(param, M2C, D_amp=(1e-9)):
    # create the Telescope object
    tel = Telescope(resolution=param['resolution'],
                    diameter=param['diameter'],
                    samplingTime=param['samplingTime'],
                    centralObstruction=param['centralObstruction'])

    # %% -----------------------     NGS   ----------------------------------
    # create the Source object
    ngs = Source(optBand=param['opticalBand'],
                 magnitude=param['magnitude'])

    # combine the NGS to the telescope using '*' operator:
    ngs * tel

    tel.computePSF(zeroPaddingFactor=6)

    # %% -----------------------     DEFORMABLE MIRROR   ----------------------------------
    # mis-registrations object
    misReg = MisRegistration(param)
    # if no coordonates specified, create a cartesian dm
    dm = DeformableMirror(telescope=tel, \
                               nSubap=param['nSubaperture'], \
                               mechCoupling=param['mechanicalCoupling'], \
                               misReg=misReg)

    wfs = TRM_Pyramid(nSubap=param['nSubaperture'],
                      telescope=tel,
                      modulation=param['modulation'],
                      lightRatio=param['lightThreshold'],
                      pupilSeparationRatio=param['pupilSeparationRatio'],
                      calibModulation=param['calibrationModulation'],
                      psfCentering=param['psfCentering'],
                      edgePixel=param['edgePixel'],
                      extraModulationFactor=param['extraModulationFactor'],
                      postProcessing=param['postProcessing'],
                      nTheta_user_defined=param["nTheta_user_defined"],
                      temporal_weights_settings=None)

    W = np.zeros((M2C.shape[1], wfs.nTheta))
    # D = np.zeros((np.sum(wfs.validSignal), bases.shape[0]))
    # Start by making the flat (no distortion image)
    dm.coefs = 0
    tel * dm * wfs
    cube_ref_no_remove = wfs.cam.cube

    # Iterate through the bases
    for b in tqdm(range(M2C.shape[1])):
        # Get phase mask on dm
        a = np.zeros(M2C.shape[1])
        a[b] = D_amp
        dm.coefs = np.matmul(M2C, a)
        tel * dm * wfs
        # Grab image
        cube_no_remove_p = wfs.cam.cube

        weights = calcDeltaIPerFrame(cube=cube_no_remove_p, cube_ref=cube_ref_no_remove, amp=D_amp)
        weights = weights / np.max(weights)

        W[b, :] = weights

        # Remove distortion
        dm.coefs = 0
        tel * dm * wfs

    return W

def generateInteractionMatrixWithFrameMask(atm, tel, wfs, bases, masks, D_amp=(1e-9)):
    D = np.zeros((wfs.cam.resolution * wfs.cam.resolution, bases.shape[0]))
    # Start by making the flat (no distortion image)
    tel-atm
    tel*wfs
    cube_ref_no_mask = wfs.cam.cube

    # Iterate through the bases
    for b in range(bases.shape[0]):
        # Get phase mask on atmosphere
        atm.OPD_no_pupil = bases[b,:,:] * D_amp
        atm.OPD = atm.OPD_no_pupil * tel.pupil
        # Apply atmosphere to telescope
        tel+atm
        # Propagate to wfs
        tel*wfs
        # Grab image
        cube_no_mask = wfs.cam.cube

        # Apply mask to each frame for this base
        ref = cube_ref_no_mask * masks[b,:, np.newaxis]
        img = cube_no_mask * masks[b, :, np.newaxis]

        # Add all frames together to get the integrated image
        flat_ref = np.sum(ref, axis=0)
        flat = np.sum(img, axis=0)

        # Calculate the difference between the normalized image and normalized reference
        I = (flat / np.sum(flat)) - (flat_ref / np.sum(flat_ref))
        I_flat = I.flatten()

        D[:,b] = I_flat / D_amp

        #Remove distortion
        tel - atm
    return D

def generateInteractionMatrix_Weighted(atm, tel, wfs, bases, removed_frames=None, amplitude=(30*1e-9)):
    """
    Generate the interaction matrix for the system. uses the bases provided
    Parameters
    ----------
    atm : atmosphere (refer to temp_atm or AO_modules.Atmosphere)
    tel : Telescope object (refer to AO_modules.Telescope)
    wfs : Wavefront sensor object (refer to AO_modules.Pyramid)
    bases : Bases to be used for the system. (num_of_bases, res, res)
    removed_frames : list of the indices of the frames to be removed

    Returns
    -------
    The interaction matrix (res^2, num_of_bases)
    """

    if removed_frames is None:
        removed_frames = []

    # if isinstance(removed_frames[0], list):
    #     D = np.zeros((len(removed_frames), wfs.cam.resolution * wfs.cam.resolution, bases.shape[0]))
    # else:
    D = np.zeros((wfs.cam.resolution * wfs.cam.resolution, bases.shape[0]))
    W = np.zeros((wfs.nTheta, bases.shape[0]))
    #D = np.zeros((np.sum(wfs.validSignal), bases.shape[0]))
    # Start by making the flat (no distortion image)
    tel-atm
    tel*wfs
    cube_ref_no_remove = wfs.cam.cube

    # Iterate through the bases
    for b in range(bases.shape[0]):
        # Get phase mask on atmosphere
        atm.OPD_no_pupil = bases[b,:,:] * amplitude
        atm.OPD = atm.OPD_no_pupil * tel.pupil
        # Apply atmosphere to telescope
        tel+atm
        # Propagate to wfs
        tel*wfs
        # Grab image
        cube_no_remove = wfs.cam.cube

        weights = calcDeltaIPerFrame(cube=cube_no_remove, cube_ref=cube_ref_no_remove, amp=amplitude)
        weights = weights / np.max(weights)

        W[:, b] = weights

        # if b == 1:
        #     plt.figure()
        #     plt.plot(weights)
        #     plt.show(block=True)

        # if isinstance(removed_frames[0], list):
        #     for i in range(len(removed_frames)):
        #         ref = np.delete(cube_ref_no_remove, removed_frames[i], axis=0)
        #         img = np.delete(cube_no_remove, removed_frames[i], axis=0)
        #
        #         flat_ref = np.sum(ref, axis=0)
        #         flat = np.sum(img, axis=0)
        #
        #         I = (flat / np.sum(flat)) - (flat_ref / np.sum(flat_ref))
        #         I_flat = I.flatten()
        #
        #         D[i, :, b] = I_flat / amplitude
        # else:
        ref = np.delete(cube_ref_no_remove, removed_frames, axis=0)
        img = np.delete(cube_no_remove, removed_frames, axis=0)

        ref = ref * weights[:, np.newaxis, np.newaxis]
        img = img * weights[:, np.newaxis, np.newaxis]


        flat_ref = np.sum(ref, axis=0)
        flat = np.sum(img, axis=0)

        I = (flat / np.sum(flat)) - (flat_ref / np.sum(flat_ref))
        I_flat = I.flatten()
        #I_flat = I[np.where(wfs.validSignal==1)]

        D[:, b] = I_flat / amplitude

        #Remove distortion
        tel - atm
    return D, W

def calcBDiag(atm, tel, wfs, bases, removed_frames=[], D_amp=(30*1e-9), mode=None):
    """
    Calculate the diagonal of B=D_plus(D_plus)^T . This corresponds to the noise factor of the system depending on the
    mode (column).
    Parameters
    ----------
    atm : atmosphere (refer to temp_atm or AO_modules.Atmosphere)
    tel : Telescope object (refer to AO_modules.Telescope)
    wfs : Wavefront sensor object (refer to AO_modules.Pyramid)
    bases : Bases to be used for the system.
    removed_frames : list of the indices of the frames to be removed

    Returns
    -------
    The diagonal of B=D_plus(D_plus)^T (num_of_bases,num_of_bases)
    """
    if mode == "binary":
        removed_frames = findOptimalFramesRemovedPerMode(tel, wfs, wfs.nTheta, bases.shape[0])
        D = generateCustomInteractionMatrix(atm, tel, wfs, bases, numRemFramesPerBase=removed_frames, D_amp=D_amp)
        D_plus = inv(D.T @ D) @ D.T
        B = D_plus @ D_plus.T

        frame_mask_settings = np.ones((wfs.nTheta, bases.shape[0]))
        for b in range(bases.shape[0]):
            frame_numbers = calcEquidistantFrameIndices(int(removed_frames[b]), wfs.nTheta)
            frame_mask_settings[frame_numbers, b] = 0

    elif mode == "weighted":
        D_w, frame_mask_settings = generateInteractionMatrix_Weighted(atm=atm, tel=tel, wfs=wfs, bases=bases, removed_frames=None,
                                                    amplitude=D_amp)
        D_w_plus = inv(D_w.T @ D_w) @ D_w.T
        B = D_w_plus @ D_w_plus.T

    else:
        D = generateInteractionMatrix(atm, tel, wfs, bases, removed_frames=removed_frames, amplitude=D_amp)
        D_plus = inv(D.T @ D) @ D.T
        B = D_plus @ D_plus.T

        frame_mask_settings = np.ones((wfs.nTheta, bases.shape[0]))
        for b in range(bases.shape[0]):
            frame_mask_settings[removed_frames, b] = 0


    return np.diag(B), frame_mask_settings


def calcPerformance(atm, tel, wfs, bases, removed_frames=None, display=True):
    B_diag_flat = calcBDiag(atm, tel, wfs, bases, removed_frames=[])
    B_diag_adj = calcBDiag(atm, tel, wfs, bases, removed_frames=removed_frames)

    if display:
        fig = plt.figure(4, figsize=(17, 10))
        ax = plt.subplot(1, 3, 1)
        ax.bar(np.arange(len(B_diag_flat)), B_diag_flat, label="No frames removed", color='b')
        ax.bar(np.arange(len(B_diag_adj)), B_diag_adj, label="With frames removed", color='tab:orange')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        ax.set_xlabel("Mode")
        ax.set_ylabel("Noise factor (unitless?)")
        ax.set_title("Noise factor as a function of the mode")
        ax.legend()

        ax2 = plt.subplot(1, 3, 2)

        ax2.plot((B_diag_adj / B_diag_flat) * 100, label="With frames removed", color='tab:orange')
        ax2.axhline(100, lw=3, linestyle='--', color='k', label="No frames removed")
        ax2.set_xlabel("Mode number")
        ax2.set_ylabel("Noise factor as % of no face removed")
        ax2.set_title('Percentage of the noise factor relative to \n no frames removed as a function of the mode')

        ax3 = plt.subplot(1, 3, 3, projection='polar')

        selected = np.delete(wfs.thetaModulation, removed_frames)
        removed = wfs.thetaModulation[removed_frames]
        colorPie = ['b'] * len(wfs.thetaModulation)
        frame_selector = ax3.bar(wfs.thetaModulation, 1, width=(2 * np.pi / wfs.thetaModulation.shape[0]), color=colorPie,
                                 edgecolor='k', picker=1)
        ax3.bar(removed, 1, width=(2 * np.pi / wfs.thetaModulation.shape[0]), color="k")

        ax3.set_title("Visualisation of the removed frames")

    return B_diag_flat, B_diag_adj


def calcEquidistantFrameIndices(num_per_face, maxFrames):
    if (maxFrames%4) != 0:
        print("Number of frames is not a multiple of 4... Aborting")
        return []
    if num_per_face == 0:
        return []

    per_face = maxFrames//4
    middle = int(per_face//2)

    frames = np.array([], dtype=int)
    for i in range(num_per_face):
        if i == 0:
            middles = np.arange(4) * per_face + middle
        else:
            f = int(((-1)**(i-1)) * (((i-1)//2)+1))
            middles = np.arange(4) * per_face + middle+f
        frames = np.append(frames, middles)

    frames.sort()
    return frames

def getPyramidImageWithDistortion(atm, tel, wfs, distortion, removed_frames=None, noise=False, display=False, seed=0, forceRefCube=None, forceCube=None):

    if removed_frames is None:
        removed_frames = []

    if forceRefCube is None:
        # Start by making the flat (no distortion image)
        tel - atm
        tel * wfs
        cube_ref_no_remove = wfs.cam.cube
    else:
        cube_ref_no_remove = forceRefCube

    cube_ref = np.delete(cube_ref_no_remove, removed_frames, axis=0)

    if forceCube is None:
        # Get phase mask on atmosphere
        atm.OPD_no_pupil = distortion
        atm.OPD = atm.OPD_no_pupil * tel.pupil
        # Apply atmosphere to telescope
        tel + atm
        # Propagate to wfs
        tel * wfs
        # Grab image
        cube_no_remove = wfs.cam.cube
    else:
        cube_no_remove = forceCube

    cube = np.delete(cube_no_remove, removed_frames, axis=0)

    flat_ref = np.sum(cube_ref, axis=0)

    flat = np.sum(cube, axis=0)
    if noise:
        #flat = random_noise(flat, mode="poisson")
        # noise_mask = np.random.poisson(flat)
        # flat = noise_mask #flat + noise_mask

        rs = np.random.RandomState(seed=seed)
        flat = rs.poisson(flat)

    if display:
        fig, _ax = plt.subplots(2,3)
        fig.subplots_adjust(hspace=0.3)
        ax = _ax.flatten()

        im0 = ax[0].imshow(tel.OPD)
        plt.colorbar(im0, ax=ax[0])
        ax[0].set_title("Telescope OPD")
        im1 = ax[1].imshow(tel.src.phase)
        plt.colorbar(im1, ax=ax[1])
        ax[1].set_title("2D map of the phase scaled to the src wavelength \n src wavelength = " + str(tel.src.wavelength))
        tel.computePSF(zeroPaddingFactor = 6)
        #im2 = ax[2].imshow(tel.PSF)
        im2 = ax[2].imshow((np.log10(tel.PSF)), extent=[tel.xPSF_arcsec[0], tel.xPSF_arcsec[1], tel.xPSF_arcsec[0], tel.xPSF_arcsec[1]])
        im2.set_clim(-1, 3)
        ax[2].set_xlabel('[Arcsec]')
        ax[2].set_ylabel('[Arcsec]')
        plt.colorbar(im2, ax=ax[2])
        ax[2].set_title("Telescope PSF")

        im3 = ax[3].imshow(flat_ref)
        plt.colorbar(im3,ax=ax[3])
        ax[3].set_title("WFS reference image,\n Photons=" + str(np.sum(flat_ref)))
        im4 = ax[4].imshow(flat)
        plt.colorbar(im4, ax=ax[4])
        ax[4].set_title("WFS distorted image with " + str(len(removed_frames)) + " removed frames,\n Photons=" + str(np.sum(flat)))

    #t = np.sum(flat)
    #t_ref = np.sum(flat_ref)
    #print(f"Photons in the image: {t}")
    #print(f"Photons in the ref: {t_ref}")
    I = (flat / np.sum(flat)) - (flat_ref / np.sum(flat_ref))

    if forceCube is None:
        # Remove distortion
        tel - atm

    return I, cube_no_remove, cube_ref_no_remove


def calcReconstructionError(atm, tel, wfs, bases, removed_frames, D_amp=(1e-9), a_amp=(1e-9), display=False,noise=False, seed=0, force_a=None, force_Dplus=None,forceRefCube=None, forceCube=None ):


    if force_Dplus is None:
        with silence():
            D = generateInteractionMatrix(atm, tel, wfs, bases, removed_frames=removed_frames, amplitude=D_amp)
        D_plus = inv(D.T @ D) @ D.T
    else:
        D_plus = force_Dplus

    if force_a is None:
        """Random 'a' vector"""
        # Perfer gaussian distribution here
        np.random.seed(seed) #TODO Validate that setting the same seed for a and the noise to not impact performance
        a = (np.random.randn(bases.shape[0], ))  # [-1,1]
        a = a * a_amp
    else:
        a = force_a * a_amp


    """The distortion associated to the 'a' vector"""
    Phi_distortion = np.zeros(bases.shape)
    for i in range(len(a)):
        Phi_distortion[i, :, :] = a[i] * bases[i, :, :]

    Phi_distortion = np.sum(Phi_distortion, axis=0)

    """Calc images with distortions"""
    with silence():
        P_img, cube, cube_ref = getPyramidImageWithDistortion(atm, tel, wfs, Phi_distortion, removed_frames=removed_frames, noise=noise, display=display,
                                            seed=seed, forceRefCube=forceRefCube, forceCube=forceCube)
    P = P_img[np.where(wfs.validSignal == 1)]
    # P = P_img.flatten()

    """Get back the 'a' vector from the pyramid image"""
    a_est = D_plus @ P

    """Error of the 'a_est' vector over the real 'a' """
    a_err = np.sqrt(np.sum((a_est - a) ** 2))

    return a, a_est, a_err, D_plus, cube ,cube_ref


def calcReconstructionErrorWithDM(dm, tel, wfs, M2C, removed_frames, D_amp=(1e-9), a_amp=(1e-9), display=False,noise=False, seed=0, force_a=None, force_Dplus=None,forceRefCube=None, forceCube=None ):


    if force_Dplus is None:
        with silence():
            D = generateInteractionMatrix(atm, tel, wfs, bases, removed_frames=removed_frames, amplitude=D_amp)
        D_plus = inv(D.T @ D) @ D.T
    else:
        D_plus = force_Dplus

    if force_a is None:
        """Random 'a' vector"""
        # Perfer gaussian distribution here
        np.random.seed(seed) #TODO Validate that setting the same seed for a and the noise to not impact performance
        a = (np.random.randn(bases.shape[0], ))  # [-1,1]
        a = a * a_amp
    else:
        a = force_a * a_amp


    """The distortion associated to the 'a' vector"""
    Phi_distortion = np.zeros(bases.shape)
    for i in range(len(a)):
        Phi_distortion[i, :, :] = a[i] * bases[i, :, :]

    Phi_distortion = np.sum(Phi_distortion, axis=0)

    """Calc images with distortions"""
    with silence():
        P_img, cube, cube_ref = getPyramidImageWithDistortion(atm, tel, wfs, Phi_distortion, removed_frames=removed_frames, noise=noise, display=display,
                                            seed=seed, forceRefCube=forceRefCube, forceCube=forceCube)
    P = P_img.flatten()

    """Get back the 'a' vector from the pyramid image"""
    a_est = D_plus @ P

    """Error of the 'a_est' vector over the real 'a' """
    a_err = np.sqrt(np.sum((a_est - a) ** 2))

    return a, a_est, a_err, D_plus, cube ,cube_ref

def calcReconstructionErrorMultiD(atm, tel, wfs, bases, numRemFramesPerBase, D_amp=(1e-9), a_amp=(1e-9), display=False,noise=False, seed=0, force_a=None, force_Dplus=None, forceRefCube=None, forceCube=None  ):


    if force_Dplus is None:
        with silence():
            remFrames = np.unique(numRemFramesPerBase)
            D_all = [None]*len(remFrames)
            for i in range(len(remFrames)):
                removed_frames = calcEquidistantFrameIndices(int(remFrames[i]), wfs.nTheta)
                D = generateInteractionMatrix(atm, tel, wfs, bases, removed_frames=removed_frames, amplitude=D_amp)
                D_all[i] = D
            D_custom = np.zeros(D.shape)
            for j in range(bases.shape[0]):
                d_idx = np.argwhere(remFrames == numRemFramesPerBase[j])[0][0]
                D_custom[:,j] = D_all[d_idx][:,j]

            D_plus = inv(D_custom.T @ D_custom) @ D_custom.T
    else:
        D_plus = force_Dplus

    if force_a is None:
        """Random 'a' vector"""
        # Perfer gaussian distribution here
        np.random.seed(seed) #TODO Validate that setting the same seed for a and the noise to not impact performance
        a = (np.random.randn(bases.shape[0], ))  # [-1,1]
        a = a * a_amp
    else:
        a = force_a * a_amp


    """The distortion associated to the 'a' vector"""
    Phi_distortion = np.zeros(bases.shape)
    for i in range(len(a)):
        Phi_distortion[i, :, :] = a[i] * bases[i, :, :]

    Phi_distortion = np.sum(Phi_distortion, axis=0)

    """Calc images with distortions"""
    with silence():
        remFrames = np.unique(numRemFramesPerBase)
        P_all = [None] * len(remFrames)
        for i in range(len(remFrames)):
            removed_frames = calcEquidistantFrameIndices(int(remFrames[i]), wfs.nTheta)
            P_img, cube, cube_ref = getPyramidImageWithDistortion(atm, tel, wfs, Phi_distortion, removed_frames=removed_frames, noise=noise, display=display,
                                        seed=seed, forceRefCube=forceRefCube, forceCube=forceCube)
            P = P_img[np.where(wfs.validSignal == 1)]
            P_all[i] = P.flatten()

    """Get back the 'a' vector from the pyramid image"""

    #a_est = D_plus @ P

    a_est = np.ones((bases.shape[0]))

    for i in range(len(a_est)):
        d_idx = np.argwhere(remFrames == numRemFramesPerBase[i])[0][0]
        a_est[i] = (D_plus @ P_all[d_idx])[i]

    """Error of the 'a_est' vector over the real 'a' """
    a_err = np.sqrt(np.sum((a_est - a) ** 2))

    return a, a_est, a_err, D_plus


def zernumero(zn):
    j=0
    for n in range(100):
        for m in range(n+1):
            if (((n-m) % 2) == 0):
                j = j+1
                if j == zn:
                    return n,m
                if m != 0 :
                    j = j+1
                    if j == zn :
                        return n,m

def generateA(length, totalDistortion):
    a = (np.random.randn(length, ))
    for i in range(length):
        order,_ = zernumero(i+2)
        a[i] = a[i] / order
    RSS = np.sqrt(np.sum(a**2))
    a = a/RSS * totalDistortion
    return a



def calcDeltaI(cube, cube_ref, removed_frames=None):

    if removed_frames is None:
        removed_frames = []

    cube_ref_removed = np.delete(cube_ref, removed_frames, axis=0)
    cube_removed = np.delete(cube, removed_frames, axis=0)

    # delta_I = np.zeros((cube_removed.shape[0]))

    # for i in range(cube_removed.shape[0]):
    #     delta_I[i] = np.sqrt(np.sum(((cube_removed[i, :, :] / np.sum(cube_removed[i, :, :])) - (cube_ref_removed[i, :, :] / np.sum(cube_ref_removed[i, :, :]))) ** 2))


    delta_I = np.sqrt(np.sum(((np.sum(cube_removed,axis=0) / np.sum(cube_removed)) - (np.sum(cube_ref_removed,axis=0) / np.sum(cube_ref_removed))) ** 2))
    #delta_I = np.sum(((np.sum(cube_removed,axis=0) / np.sum(cube_removed)) - (np.sum(cube_ref_removed,axis=0) / np.sum(cube_ref_removed))))


    return delta_I

def calcDeltaIPerFrame(cube, cube_ref,amp=1.0):
    delta_I = np.zeros((cube.shape[0]))
    for i in range(cube.shape[0]):
        delta_I[i] = np.sqrt( np.sum((((cube[i, :, :] / np.sum(cube[i, :, :])) - (cube_ref[i, :, :] / np.sum(cube_ref[i, :, :])))/amp) ** 2))

    return delta_I


@contextlib.contextmanager
def silence():
    sys.stdout, old = io.StringIO(), sys.stdout
    try:
        yield
    finally:
        sys.stdout = old

def dict_to_trajectory(d, traj):
    for k,v in d.items():
        traj.f_add_parameter(k, v, comment=k)

def generateCustomInteractionMatrix(atm, tel, wfs, bases, numRemFramesPerBase, D_amp=(1e-9)):
    remFrames = np.unique(numRemFramesPerBase)
    D_all = [None] * len(remFrames)
    for i in range(len(remFrames)):
        removed_frames = calcEquidistantFrameIndices(int(remFrames[i]), wfs.nTheta)
        D = generateInteractionMatrix(atm, tel, wfs, bases, removed_frames=removed_frames, amplitude=D_amp)
        D_all[i] = D
    D_custom = np.zeros(D.shape)
    for j in range(bases.shape[0]):
        d_idx = np.argwhere(remFrames == numRemFramesPerBase[j])[0][0]
        D_custom[:, j] = D_all[d_idx][:, j]

    D_plus = inv(D_custom.T @ D_custom) @ D_custom.T

    return D_plus


def generateCustomInteractionMatrixFromDM(dm, tel, wfs, M2C, numRemFramesPerBase, D_amp=(1e-9)):
    remFrames = np.unique(numRemFramesPerBase)
    D_all = [None] * len(remFrames)
    for i in range(len(remFrames)):
        removed_frames = calcEquidistantFrameIndices(int(remFrames[i]), wfs.nTheta)
        D = generateInteractionMatrixWithDM_onlyPOS(dm, tel, wfs, M2C, removed_frames=removed_frames, amplitude=D_amp)
        D_all[i] = D
    D_custom = np.zeros(D.shape)
    for j in range(M2C.shape[1]):
        d_idx = np.argwhere(remFrames == numRemFramesPerBase[j])[0][0]
        D_custom[:, j] = D_all[d_idx][:, j]

    D_plus = inv(D_custom.T @ D_custom) @ D_custom.T

    return D_plus, D_custom


def generateInteractionMatrixWithDM_onlyPOS(dm, tel, wfs, C2M, removed_frames=None, amplitude=(1e-9)):
    if removed_frames is None:
        removed_frames = []

    # if isinstance(removed_frames[0], list):
    #     D = np.zeros((len(removed_frames), wfs.cam.resolution * wfs.cam.resolution, bases.shape[0]))
    # else:
    D = np.zeros((wfs.validSignal.sum(), C2M.shape[1]))
    #D = np.zeros((wfs.cam.resolution * wfs.cam.resolution, C2M.shape[1]))
    # Start by making the flat (no distortion image)
    dm.coefs = 0
    tel * dm * wfs
    cube_ref_no_remove = wfs.cam.cube

    # Iterate through the bases
    for b in tqdm(range(C2M.shape[1])):
        # Get phase mask on dm
        a = np.zeros(C2M.shape[1])
        a[b] = amplitude
        dm.coefs = np.matmul(C2M, a)
        tel * dm * wfs
        # Grab image
        cube_no_remove_p = wfs.cam.cube

        ref = np.delete(cube_ref_no_remove, removed_frames, axis=0)
        img_p = np.delete(cube_no_remove_p, removed_frames, axis=0)


        flat_ref = np.sum(ref, axis=0)
        flat_p = np.sum(img_p, axis=0)

        flat_ref = flat_ref[np.where(wfs.validSignal == 1)]
        flat_p   =   flat_p[np.where(wfs.validSignal == 1)]


        I_p = (flat_p / np.sum(flat_p)) - (flat_ref / np.sum(flat_ref))
        # I_flat_p = I_p.flatten()

        D[:, b] = I_p / amplitude

        #Remove distortion
        dm.coefs = 0
        tel * dm * wfs
    return D


def generateInteractionMatrixWithDM(dm, tel, wfs, C2M, removed_frames=None, amplitude=(1e-9)):
    if removed_frames is None:
        removed_frames = []

    # if isinstance(removed_frames[0], list):
    #     D = np.zeros((len(removed_frames), wfs.cam.resolution * wfs.cam.resolution, bases.shape[0]))
    # else:
    D = np.zeros((wfs.validSignal.sum(), C2M.shape[1]))
    # Start by making the flat (no distortion image)
    dm.coefs = 0
    tel * dm * wfs
    cube_ref_no_remove = wfs.cam.cube

    # Iterate through the bases
    for b in tqdm(range(C2M.shape[1])):
        # Get phase mask on dm
        dm.coefs = C2M[:,b] * amplitude
        tel * dm * wfs
        # Grab image
        cube_no_remove_p = wfs.cam.cube

        # dm.coefs = 0
        # tel * dm * wfs

        dm.coefs = C2M[:, b] * -amplitude
        tel * dm * wfs
        # Grab image
        cube_no_remove_n = wfs.cam.cube

        ref = np.delete(cube_ref_no_remove, removed_frames, axis=0)
        img_p = np.delete(cube_no_remove_p, removed_frames, axis=0)
        img_n = np.delete(cube_no_remove_n, removed_frames, axis=0)

        flat_ref = np.sum(ref, axis=0)
        flat_p = np.sum(img_p, axis=0)
        flat_n = np.sum(img_n, axis=0)

        flat_ref = flat_ref[np.where(wfs.validSignal == 1)]
        flat_p   =   flat_p[np.where(wfs.validSignal == 1)]
        flat_n   =   flat_n[np.where(wfs.validSignal == 1)]

        I_p = (flat_p / np.sum(flat_p)) - (flat_ref / np.sum(flat_ref))
        I_flat_p = I_p.flatten()

        I_n = (flat_n / np.sum(flat_n)) - (flat_ref / np.sum(flat_ref))
        I_flat_n = I_n.flatten()

        D[:, b] = 0.5 * (I_flat_p-I_flat_n) / amplitude


        #Remove distortion
        dm.coefs = 0
        tel * dm * wfs
    return D

def TRreconstructor(wfs, refCube, numRemFramesPerBase, C2M, CL):
    if wfs is None:
        return np.zeros((C2M.shape[0]))

    a_est = np.ones((CL.shape[0]))

    imageCube = wfs.cam.cube

    remFrames = np.unique(numRemFramesPerBase)
    P_all = np.zeros((len(remFrames), CL.shape[1]))

    for i in range(len(remFrames)):
        removeFramesIdx = calcEquidistantFrameIndices(int(remFrames[i]), wfs.nTheta)
        t_cube= np.delete(imageCube, removeFramesIdx, axis=0)
        t_refcube = np.delete(refCube, removeFramesIdx, axis=0)
        I = np.sum(t_cube, axis=0)/ np.sum(t_cube) - np.sum(t_refcube, axis=0)/ np.sum(t_refcube)
        P_all[i,:] = I.flatten()


    for i in range(len(a_est)):
        d_idx = np.argwhere(remFrames == numRemFramesPerBase[i])[0][0]
        a_est[i] = (CL @ P_all[d_idx,:])[i]

    out = C2M@a_est

    return out


def paramSummary(param):
    data = {"Atmosphere r0": param['r0'],
            "Telescope diameter (m)": param['diameter'],
            "nSubaperture": param['nSubaperture'],
            "WFS sampling time (s)": param['samplingTime'],
            "Object magnitude": param['magnitude'],
            "Object optical band": param['opticalBand'],
            "WFS modulation (lambda)": param['modulation']}
    def formatLine(field, value, fieldLength=10):
        return f"{field:>{fieldLength}} :: {value:>5}\n"
    longestKey = len(max(list(data.keys()),key=len))

    text = ""
    for k,v in data.items():
        text += formatLine(k,v,fieldLength=longestKey)

    return text



def findOptimalFramesRemovedPerMode(tel, wfs, nTheta_user_defined, numBases):

    if (((nTheta_user_defined // 4) % 2) == 0):
        REMOVED_FRAMES_SETTINGS = np.array([0])
        frms = np.array(list(range((nTheta_user_defined // 4))))
        REMOVED_FRAMES_SETTINGS = np.append(REMOVED_FRAMES_SETTINGS, frms[frms % 2 == 1])  # [0,1,3,5,7,9,11]
    else:
        frms = np.array(list(range((nTheta_user_defined // 4))))
        REMOVED_FRAMES_SETTINGS = frms[frms % 2 == 0]


    B_diags = np.zeros((len(REMOVED_FRAMES_SETTINGS), numBases))
    D_amp = 30*(1e-9)

    atm = dummy_atm(tel)

    bases = generateBases(numBases, tel.resolution, baseType="KL", display=False, scale=False)

    for i in range(len(REMOVED_FRAMES_SETTINGS)):
        removed_frames = calcEquidistantFrameIndices(REMOVED_FRAMES_SETTINGS[i], wfs.nTheta)
        B_diags[i, :], _ = calcBDiag(atm, tel, wfs, bases, removed_frames=removed_frames, D_amp=D_amp)


    factor = np.zeros(B_diags.shape)
    for i in range(0, len(REMOVED_FRAMES_SETTINGS)):
        factor[i, :] = ((nTheta_user_defined - (REMOVED_FRAMES_SETTINGS[i] * 4)) / nTheta_user_defined) / B_diags[i, :]

    best_setting = [0] * numBases
    for j in range(numBases):
        best_setting[j] = REMOVED_FRAMES_SETTINGS[np.argmax(factor[:, j])]

    return best_setting

def findOptimalFrames(tel, wfs, nTheta_user_defined, bases):
    numBases = bases.shape[0]

    if (((nTheta_user_defined // 4) % 2) == 0):
        REMOVED_FRAMES_SETTINGS = np.array([0])
        frms = np.array(list(range((nTheta_user_defined // 4))))
        REMOVED_FRAMES_SETTINGS = np.append(REMOVED_FRAMES_SETTINGS, frms[frms % 2 == 1])  # [0,1,3,5,7,9,11]
    else:
        frms = np.array(list(range((nTheta_user_defined // 4))))
        REMOVED_FRAMES_SETTINGS = frms[frms % 2 == 0]


    B_diags = np.zeros((len(REMOVED_FRAMES_SETTINGS), numBases))
    D_amp = (1e-9)

    atm = dummy_atm(tel)


    for i in range(len(REMOVED_FRAMES_SETTINGS)):
        removed_frames = calcEquidistantFrameIndices(REMOVED_FRAMES_SETTINGS[i], wfs.nTheta)
        B_diags[i, :], _ = calcBDiag(atm, tel, wfs, bases, removed_frames=removed_frames, D_amp=D_amp)


    factor = np.zeros(B_diags.shape)
    for i in range(0, len(REMOVED_FRAMES_SETTINGS)):
        factor[i, :] = ((nTheta_user_defined - (REMOVED_FRAMES_SETTINGS[i] * 4)) / nTheta_user_defined) / B_diags[i, :]

    best_setting = [0] * numBases
    for j in range(numBases):
        best_setting[j] = REMOVED_FRAMES_SETTINGS[np.argmax(factor[:, j])]

    return best_setting

def calcReconstructionErrorWithWeights(atm, tel, wfs, bases, weights, D_amp=(1e-9), a_amp=(1e-9), display=False,noise=False, seed=0, force_a=None, force_Dplus=None, forceRefCube=None, forceCube=None  ):

    if force_Dplus is None:
        with silence():
            D_custom, weights = generateInteractionMatrix_Weighted(atm, tel, wfs, bases, removed_frames=None, amplitude=D_amp)
            #D_custom = generateInteractionMatrix(atm, tel, wfs, bases, removed_frames=None, amplitude=D_amp)
            D_plus = inv(D_custom.T @ D_custom) @ D_custom.T
    else:
        D_plus = force_Dplus

    if force_a is None:
        """Random 'a' vector"""
        # Perfer gaussian distribution here
        np.random.seed(seed) #TODO Validate that setting the same seed for a and the noise to not impact performance
        a = (np.random.randn(bases.shape[0], ))  # [-1,1]
        a = a * a_amp
    else:
        a = force_a * a_amp


    """The distortion associated to the 'a' vector"""
    Phi_distortion = np.zeros(bases.shape)
    for i in range(len(a)):
        Phi_distortion[i, :, :] = a[i] * bases[i, :, :]

    Phi_distortion = np.sum(Phi_distortion, axis=0)

    """Calc images with distortions"""
    with silence():

        P_all = [None] * len(bases)
        for i in range(len(bases)):
            P_img, cube, cube_ref = getPyramidImageWithDistortion(atm, tel, wfs, Phi_distortion,
                                                                  removed_frames=None, noise=noise,
                                                                  display=display,
                                                                  seed=seed, forceRefCube=forceRefCube, forceCube=forceCube)

            rs = np.random.RandomState(seed=seed)
            cube_n = rs.poisson(cube)
            ref_w = cube_ref * weights[:, i, np.newaxis, np.newaxis]
            cube_w = cube_n * weights[:, i, np.newaxis, np.newaxis]
            # ref_w = cube_ref
            # cube_w = cube

            # if i == 1:
            #     plt.figure()
            #     plt.plot(weights[:, i])
            #     plt.show(block=True)


            flat_ref = np.sum(ref_w, axis=0)
            flat = np.sum(cube_w, axis=0)



            P_img_w = (flat / np.sum(flat)) - (flat_ref / np.sum(flat_ref))
            P_all[i] = P_img_w.flatten()

    """Get back the 'a' vector from the pyramid image"""


    a_est = np.ones((bases.shape[0]))

    for i in range(len(a_est)):
        a_est[i] = (D_plus @ P_all[i])[i]


    """Error of the 'a_est' vector over the real 'a' """
    a_err = np.sqrt(np.sum((a_est - a) ** 2))

    return a, a_est, a_err, D_plus, weights

