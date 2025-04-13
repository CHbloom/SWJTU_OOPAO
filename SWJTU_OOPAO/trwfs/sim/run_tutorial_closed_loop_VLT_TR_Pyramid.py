"""
Created on Wed Oct 21 10:51:32 2020

@author: cheritie
"""

import time

import matplotlib.pyplot as plt

import numpy as np

from OOPAO.Atmosphere import Atmosphere
from OOPAO.DeformableMirror import DeformableMirror
from OOPAO.MisRegistration import MisRegistration
from OOPAO.Pyramid import Pyramid
from OOPAO.TR_Pyramid import TR_Pyramid
from OOPAO.Source import Source
from OOPAO.Telescope import Telescope
from OOPAO.Zernike import Zernike
from OOPAO.calibration.CalibrationVault import CalibrationVault
from OOPAO.calibration.InteractionMatrix import InteractionMatrix, InteractionMatrix_weightedFrames
from OOPAO.calibration.compute_KL_modal_basis import compute_M2C, compute_KL_basis

from OOPAO.tools.displayTools import cl_plot, displayMap
#-----------------------     read parameter file   ----------------------------------
from trwfs.parameter_files.parameterFile_CMOS_PWFS_aug2022_3 import initializeParameterFile

param = initializeParameterFile()


#plt.ion()

# -----------------------     TELESCOPE   ----------------------------------

# create the Telescope object
tel = Telescope(resolution          = param['resolution'],\
                diameter            = param['diameter'],\
                samplingTime        = param['samplingTime'],\
                centralObstruction  = param['centralObstruction'])

# -----------------------     NGS   ----------------------------------
# create the Source object
ngs=Source(optBand   = param['opticalBand'],\
           magnitude = param['magnitude'])

# combine the NGS to the telescope using '*' operator:
ngs*tel

tel.computePSF(zeroPaddingFactor = 6)
plt.figure()
plt.imshow(np.log10(np.abs(tel.PSF)),extent = [tel.xPSF_arcsec[0],tel.xPSF_arcsec[1],tel.xPSF_arcsec[0],tel.xPSF_arcsec[1]])
plt.clim([-1,3])
plt.xlabel('[Arcsec]')
plt.ylabel('[Arcsec]')
plt.colorbar()

src=Source(optBand   = 'K',\
           magnitude = param['magnitude'])

# -----------------------     ATMOSPHERE   ----------------------------------

# create the Atmosphere object
atm=Atmosphere(telescope     = tel,\
               r0            = param['r0'],\
               L0            = param['L0'],\
               windSpeed     = param['windSpeed'],\
               fractionalR0  = param['fractionnalR0'],\
               windDirection = param['windDirection'],\
               altitude      = param['altitude'])
# initialize atmosphere
atm.initializeAtmosphere(tel)

atm.update()

plt.figure()
plt.imshow(atm.OPD*1e9)
plt.title('OPD Turbulence [nm]')
plt.colorbar()


tel+atm
tel.computePSF(8)
plt.figure()
plt.imshow((np.log10(tel.PSF)),extent = [tel.xPSF_arcsec[0],tel.xPSF_arcsec[1],tel.xPSF_arcsec[0],tel.xPSF_arcsec[1]])
plt.clim([-1,3])

plt.xlabel('[Arcsec]')
plt.ylabel('[Arcsec]')
plt.colorbar()
tel.print_optical_path()
tel-atm
tel.print_optical_path()


plt.close('all')
tel.OPD_no_pupil = atm.OPD * 1000

plt.figure()
plt.imshow(tel.OPD*1e9)
plt.title('OPD Turbulence [nm]')
plt.colorbar()



plt.figure()
plt.imshow(tel.OPD_no_pupil*1e9)
plt.title('OPD Turbulence [nm]')
plt.colorbar()
#-----------------------     DEFORMABLE MIRROR   ----------------------------------
# mis-registrations object
misReg = MisRegistration(param)
# if no coordonates specified, create a cartesian dm
dm=DeformableMirror(telescope    = tel,\
                    nSubap       = 20,\
                    mechCoupling = param['mechanicalCoupling'],\
                    misReg       = misReg)

plt.figure()
plt.plot(dm.coordinates[:,0],dm.coordinates[:,1],'x')
plt.xlabel('[m]')
plt.ylabel('[m]')
plt.title('DM Actuator Coordinates')

tel*dm
#-----------------------     PYRAMID WFS   ----------------------------------

# make sure tel and atm are separated to initialize the PWFS
tel-atm


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
                 nTheta_user_defined = param['nTheta_user_defined'])





# -----------------------     Modal Basis   ----------------------------------
# compute the modal basis



from OOPAO.calibration.compute_KL_modal_basis import compute_KL_basis
import matplotlib.cm as cm
M2C_KL = compute_KL_basis(tel,atm,dm)


# show the first 10 KL
dm.coefs = M2C_KL[:,:10]
tel*dm
displayMap(tel.OPD)

# to manually measure the interaction matrix
# amplitude of the modes in m
stroke=1e-9


# Calculate the weights per frames per mode
tel - atm
wfs.cam.photonNoise = False
nModes = M2C_KL.shape(1)

i_cube = np.zeros((wfs.nTheta, np.sum(wfs.validSignal), nModes))

dm.coefs = 0
ngs* tel * dm * wfs
ref_cube = wfs.cam.cube[:, wfs.validSignal]

# bases = generateBases(nModes, tel.resolution, baseType="KL", display=False, scale=False)
for i in range(nModes):
    dm.coefs = M2C_KL[:, i] * stroke
    tel * dm * wfs
    push = wfs.cam.cube[:, wfs.validSignal]
    push_signal = push/np.sum(push) - \
                      ref_cube/np.sum(ref_cube)

    dm.coefs = -M2C_KL[:, i] * stroke
    tel * dm * wfs
    pull = wfs.cam.cube[:, wfs.validSignal]
    pull_signal = pull/np.sum(pull) - \
                      ref_cube/np.sum(ref_cube)

    i_cube[:,:,i] = (0.5 * (push_signal - pull_signal) / stroke)

weighting_cube = np.zeros((wfs.nTheta, nModes))
for i in range(nModes):
    avg_val = np.mean(i_cube[:, :, i])
    weighting_cube[:,i] = ((np.mean((i_cube[:, :, i]-avg_val)**2, axis=1)))**2
    weighting_cube[:,i] = weighting_cube[:,i]  / np.max(np.abs(weighting_cube[:,i]))

plt.figure()
im = plt.imshow(weighting_cube, cmap=cm.Greys)
im1_cbar = plt.colorbar(im)
plt.title("Normalized weight for each frame of the modulation with every KL mode")
plt.ylabel("Modulation Frame")
plt.xlabel("KL mode")
plt.figure()











# Modal Interaction Matrix
calib_modal = InteractionMatrix_weightedFrames(  ngs            = ngs,\
                            atm            = atm,\
                            tel            = tel,\
                            dm             = dm,\
                            wfs            = wfs,\
                            M2C            = M2C_KL,\
                            stroke         = stroke,\
                            nMeasurements  = 1,\
                            noise          = 'off',
                            custom_frames=True,
                            frame_weights=weighting_cube  )

plt.figure()
plt.plot(np.std(calib_modal.D,axis=0))
plt.xlabel('Mode Number')
plt.ylabel('WFS slopes STD')






# Modal interaction matrix
calib_KL = CalibrationVault(calib_modal.D@M2C_KL)

plt.figure()
plt.plot(np.sqrt(np.diag(calib_KL.D.T@calib_KL.D))/calib_KL.D.shape[0]/ngs.fluxMap.sum())
plt.xlabel('Mode Number')
plt.ylabel('WFS slopes STD')



tel.resetOPD()
# initialize DM commands
dm.coefs=0
ngs*tel*dm*wfs
tel+atm

# dm.coefs[100] = -1

tel.computePSF(4)
plt.close('all')
    
# These are the calibration data used to close the loop
calib_CL    = calib_KL
M2C_CL      = M2C_KL


# combine telescope with atmosphere
tel+atm

# initialize DM commands
dm.coefs=0
ngs*tel*dm*wfs

import matplotlib
matplotlib.use("TkAgg")

plt.show()

param['nLoop'] = 500
# allocate memory to save data
SR                      = np.zeros(param['nLoop'])
total                   = np.zeros(param['nLoop'])
residual                = np.zeros(param['nLoop'])
wfsSignal               = np.arange(0,wfs.nSignal)*0
SE_PSF = []
LE_PSF = np.log10(tel.PSF_norma_zoom)
SE_PSF_K = []

plot_obj = cl_plot(list_fig          = [atm.OPD,tel.mean_removed_OPD,wfs.cam.frame,np.log10(wfs.get_modulation_frame(radius = 10)),[[0,0],[0,0]],[dm.coordinates[:,0],np.flip(dm.coordinates[:,1]),dm.coefs],np.log10(tel.PSF_norma_zoom),np.log10(tel.PSF_norma_zoom)],\
                   type_fig          = ['imshow','imshow','imshow','imshow','plot','scatter','imshow','imshow'],\
                   list_title        = ['Turbulence OPD [m]','Residual OPD [m]','WFS Detector Plane','WFS Focal Plane',None,None,None,None],\
                   list_lim          = [None,None,None,[-3,0],None,None,[-4,0],[-5,0]],\
                   list_label        = [None,None,None,None,['Time [ms]','WFE [nm]'],['DM Commands',''],['Short Exposure I Band PSF',''],['Long Exposure K Band PSF','']],\
                   n_subplot         = [4,2],\
                   list_display_axis = [None,None,None,None,True,None,None,None],\
                   list_ratio        = [[0.95,0.95,0.1],[1,1,1,1]], s=20)
# loop parameters
gainCL                  = 0.6
wfs.cam.photonNoise     = True
display                 = True

reconstructor = M2C_CL@calib_CL.M

for i in range(param['nLoop']):
    a=time.time()
    # update phase screens => overwrite tel.OPD and consequently tel.src.phase
    atm.update()
    # save phase variance
    total[i]=np.std(tel.OPD[np.where(tel.pupil>0)])*1e9
    # save turbulent phase
    turbPhase = tel.src.phase
    # propagate to the WFS with the CL commands applied
    ngs*tel*dm*wfs

    test = wfs.get_modulation_frame()
        
    dm.coefs=dm.coefs-gainCL*np.matmul(reconstructor,wfsSignal)
    # store the slopes after computing the commands => 2 frames delay
    wfsSignal=wfs.signal
    b= time.time()
    print('Elapsed time: ' + str(b-a) +' s')
    # update displays if required
    if display==True:        
        tel.computePSF(4)
        SE_PSF.append(np.log10(tel.PSF_norma_zoom))

        if i>15:
            src*tel
            tel.computePSF(4)
            SE_PSF_K.append(np.log10(tel.PSF_norma_zoom))

            LE_PSF = np.mean(SE_PSF_K, axis=0)
        
        cl_plot(list_fig   = [atm.OPD,tel.mean_removed_OPD,wfs.cam.frame,np.log10(wfs.get_modulation_frame(radius=10)),[np.arange(i+1),residual[:i+1]],dm.coefs,(SE_PSF[-1]), LE_PSF],
                               plt_obj = plot_obj)
        plt.pause(0.1)
        if plot_obj.keep_going is False:
            break
    
    SR[i]=np.exp(-np.var(tel.src.phase[np.where(tel.pupil==1)]))
    residual[i]=np.std(tel.OPD[np.where(tel.pupil>0)])*1e9
    OPD=tel.OPD[np.where(tel.pupil>0)]

    print('Loop'+str(i)+'/'+str(param['nLoop'])+' Turbulence: '+str(total[i])+' -- Residual:' +str(residual[i])+ '\n')


plt.figure()
plt.plot(total)
plt.plot(residual)
plt.xlabel('Time')
plt.ylabel('WFE [nm]')

