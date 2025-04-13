import numpy as np
import matplotlib.pyplot as plt
plt.ion()
import matplotlib.cm as cm
import matplotlib.animation as animation
from trwfs.tools.PWFS_CMOS_NOISE_PROOF_may2022_tools import *
from OOPAO.Atmosphere       import Atmosphere
from OOPAO.TR_Pyramid import TR_Pyramid
from OOPAO.DeformableMirror import DeformableMirror
from OOPAO.MisRegistration  import MisRegistration
from OOPAO.Telescope        import Telescope
from OOPAO.Source           import Source
from pypet import Environment, cartesian_product
from astropy.io import fits
from matplotlib.colors import LogNorm

class PyramidPropagationSimulation:

    def __init__(self):
        pass

    def run(self, traj):
        param = traj.parameters.f_to_dict(fast_access=True, short_names=True)
        # %% -----------------------     TELESCOPE   ----------------------------------
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
        traj.f_add_result('telescope_PSF', tel.PSF, comment="PSF of the telescope")
        traj.f_add_result('telescope_xPSF_arcsec', tel.xPSF_arcsec, comment="Arcsec delimitation of the PSF")
        # %% -----------------------     ATMOSPHERE   ----------------------------------
        wavelengthPix = param['resolution'] / traj.mask_cycles
        phaseMask = fourierPhaseMask(wavelengthPix=wavelengthPix, angleRad=traj.mask_angle, size=param['resolution'])
        phaseMask = phaseMask / np.abs(np.max(phaseMask))
        phaseMask = phaseMask * traj.mask_amplitude


        atm = dummy_atm(tel)
        traj.f_add_result("wavelength", atm.wavelength, comment="The wavelength used for the simulation.")
        atm.OPD_no_pupil = phaseMask
        atm.OPD = atm.OPD_no_pupil * tel.pupil

        traj.f_add_result('phaseMask', atm.OPD, comment="Phase mask used as the distortion")
        traj.f_add_result('telPupil', tel.pupil, comment="Telescope Pupil")


        #%% -----------------------     PYRAMID WFS   ----------------------------------

        # make sure tel and atm are separated to initialize the PWFS
        tel - atm

        wfs = TR_Pyramid( nSubap                = param['nSubaperture'],
                          telescope             = tel,
                          modulation            = param['modulation'],
                          lightRatio            = param['lightThreshold'],
                          pupilSeparationRatio  = param['pupilSeparationRatio'],
                          calibModulation       = param['calibrationModulation'],
                          psfCentering          = param['psfCentering'],
                          edgePixel             = param['edgePixel'],
                          extraModulationFactor = param['extraModulationFactor'],
                          postProcessing        = param['postProcessing'],
                          nTheta_user_defined   = param['nTheta_user_defined'])

        wfs.cam.photonNoise = False

        # %% -----------------------     DATA   ----------------------------------
        tel * wfs

        cube_ref = wfs.cam.cube
        traj.f_add_result('cube_ref', cube_ref, comment="Reference image frames of the pyramid WFS (no distortion)")

        tel + atm
        tel * wfs
        cube = wfs.cam.cube
        traj.f_add_result('cube', cube, comment="Image frames of the pyramid WFS with distortion")


        image_cube_o = np.abs(wfs.modulation_camera_em)
        mod_camera_frame_phase = np.array(wfs.modulation_camera_em_phase)
        image_cube = np.zeros(image_cube_o.shape)
        for i in range(len(wfs.thetaModulation)):
            val_phase = (wfs.thetaModulation[i])
            idx = np.where(mod_camera_frame_phase == val_phase)
            image_cube[i, :, :] = np.squeeze(image_cube_o[idx, :, :])

        traj.f_add_result('image_cube', image_cube, comment="Images of the PSF during modulation")
        traj.f_add_result('thetaModulation', wfs.thetaModulation, comment="Modulation angles used for each frame of the image cube")

        delta_I = np.zeros((cube.shape[0]))
        for i in range(cube.shape[0]):
            delta_I[i] = np.sqrt(np.mean(((cube[i, :, :] / np.sum(cube[i, :, :])) - (cube_ref[i, :, :] / np.sum(cube_ref[i, :, :]))) ** 2))

        traj.f_add_result('delta_I', delta_I, comment="The RMS difference between the reference and the image. Indicates the information contained in that frame.")


    def view(self, traj, modulation, mask_cycles, mask_angle, mask_amplitude):
        filter_function = lambda modulation_l, mask_cycles_l, mask_angle_l, mask_amplitude_l: modulation_l == modulation and \
                                                                                              mask_cycles_l == mask_cycles and \
                                                                                              mask_angle_l == mask_angle and \
                                                                                              mask_amplitude_l == mask_amplitude
        idx_iterator = traj.f_find_idx(['parameters.modulation', 'parameters.mask_cycles', 'parameters.mask_angle', 'parameters.mask_amplitude'], filter_function)
        for idx in idx_iterator:
            traj.v_idx = idx

        fig = plt.figure(figsize=(15, 10))
        #anim_running = True

        ax10 = plt.subplot(3,3,1)
        ax10.set_title("Phase mask (nm)")

        phasemask_view = ax10.imshow(traj.crun.phaseMask)
        plt.colorbar(phasemask_view)

        ax11 = plt.subplot(3,3,2)
        ax11.set_title("Telescope PSF")
        telescope_view = ax11.imshow(np.log10(np.abs(traj.crun.telescope_PSF)), extent=[traj.crun.telescope_xPSF_arcsec[0], traj.crun.telescope_xPSF_arcsec[1], traj.crun.telescope_xPSF_arcsec[0], traj.crun.telescope_xPSF_arcsec[1]])
        telescope_view.set_clim(1, 5)
        ax11.set_xlabel('[Arcsec]')
        ax11.set_ylabel('[Arcsec]')
        plt.colorbar(telescope_view)

        ax12 = plt.subplot(3, 3, 3)
        run_information =  f"Modulation={modulation} lambda/D \n" \
                           f"D = {traj.diameter} m\n" \
                           f"wavelength = {traj.crun.wavelength*(1e9):.2f} nm\n" \
                           f"cycles = {mask_cycles} \n" \
                           f"angle={180*mask_angle/np.pi:.2f} deg\n" \
                           f"mask_amp ={int(mask_amplitude*(1e9))} nm \n" \
                           f"mask_amp meas. (max-min) ={int((np.max(traj.crun.phaseMask)-np.min(traj.crun.phaseMask))*(1e9))} nm \n" \
                           f"mask_amp avg. ={np.mean(traj.crun.phaseMask) * (1e9)} nm \n" \
                           f"sqrt(mean(phaseMask²)) (RMS) = {int(np.sqrt(np.mean(traj.crun.phaseMask ** 2)) * (1e9))} nm \n" \
                           f"phaseMask Pupil only RMS = {int(np.sqrt(np.sum((traj.crun.phaseMask ** 2)/np.sum(traj.crun.telPupil))) * (1e9))} nm \n" \
                           f"magnitude = {traj.magnitude}"
        ax12.set_axis_off()
        ax12.text(0.1, 0.85, run_information,horizontalalignment='left', verticalalignment='center')


        ax1 = plt.subplot(3, 3, 4)
        ax1.set_title("Aberrated\npyramid view")
        ax1_d = plt.subplot(3, 3, 5)
        ax1_d.set_title("Difference\nAber. - ref.")
        ax1_r = plt.subplot(3, 3, 6)
        ax1_r.set_title("Reference\npyramid view")
        ax2 = plt.subplot(3, 3, 7, projection='polar')
        ax2.set_title(r'Beam position and $\Delta \mathrm{I} ( \phi )$')
        ax3 = plt.subplot(3, 3, 8)
        ax3.set_title(
            '$\Delta \mathrm{I} ( \phi )$ as a function\nof the frame number\n(max = ' + str(traj.crun.cube.shape[0]) + ')')
        ax4 = plt.subplot(3, 3, 9)
        ax4.set_title("PSF position relative\nto the pyramid peak")

        # set the spacing between subplots
        plt.subplots_adjust(left=0.1,
                            bottom=0.1,
                            right=0.9,
                            top=0.9,
                            wspace=0.4,
                            hspace=0.4)

        cube_diff = traj.crun.cube - traj.crun.cube_ref
        distance = (traj.crun.wavelength / traj['diameter']) * traj['modulation']

        ax3.bar(list(range(traj.crun.cube.shape[0])), traj.crun.delta_I)
        ax2.bar(2 * np.pi - traj.crun.thetaModulation, traj.crun.delta_I, width=(2 * np.pi / traj.crun.cube.shape[0]), edgecolor="c")

        im1 = ax1.imshow(traj.crun.cube[0, :, :], cmap=cm.Greys_r)
        im2 = ax1_r.imshow(traj.crun.cube_ref[0, :, :], cmap=cm.Greys_r)
        im3 = ax1_d.imshow(cube_diff[0, :, :], cmap=cm.Greys_r)
        im4 = ax2.scatter((2 * np.pi - traj.crun.thetaModulation[0]), distance, s=200)
        im5 = ax3.axvline(0, color='r')
        ax2.axvline(0, lw=2, alpha=0.5, color='k')
        ax2.axvline(np.pi * 0.5, lw=2, alpha=0.5, color='k')
        ax2.axvline(np.pi, lw=2, alpha=0.5, color='k')
        ax2.axvline(np.pi * 1.5, lw=2, alpha=0.5, color='k')
        im6 = ax2.axvline(0, lw=3, color='r')
        im7 = ax4.imshow(traj.crun.image_cube[0, :, :], cmap=cm.Greys_r)
        mid = traj.crun.image_cube.shape[1] / 2.0
        ax4.axvline(mid, lw=2, alpha=0.5, color='r')
        ax4.axhline(mid, lw=2, alpha=0.5, color='r')

        im1_cbar = plt.colorbar(im1, ax=ax1)
        im1_cbar.ax.set_autoscale_on(True)
        im2_cbar = plt.colorbar(im2, ax=ax1_r)
        im2_cbar.ax.set_autoscale_on(True)
        im3_cbar = plt.colorbar(im3, ax=ax1_d)
        im3_cbar.ax.set_autoscale_on(True)

        def funcAnim(i):
            """ Aberrated wavefront """
            im1.set_data(traj.crun.cube[i, :, :])
            im1.set_clim(vmin=traj.crun.cube[i, :, :].min(), vmax=traj.crun.cube[i, :, :].max())
            """ Reference wavefront """
            im2.set_data(traj.crun.cube_ref[i, :, :])
            im2.set_clim(vmin=traj.crun.cube_ref[i, :, :].min(), vmax=traj.crun.cube_ref[i, :, :].max())
            """ Difference of aberrated with reference wavefront """
            im3.set_data(cube_diff[i, :, :])
            im3.set_clim(vmin=cube_diff[i, :, :].min(), vmax=cube_diff[i, :, :].max())
            """ Beam position """
            im7.set_data(traj.crun.image_cube[i, :, :])
            im7.set_clim(vmin=traj.crun.image_cube[i, :, :].min(), vmax=traj.crun.image_cube[i, :, :].max())

            ax4.set_xlim(mid - 50, mid + 50)
            ax4.set_ylim(mid - 50, mid + 50)
            im4.set_offsets([(2 * np.pi - traj.crun.thetaModulation[i]), distance])
            im5.set_xdata(list(range(len(traj.crun.thetaModulation)))[i])
            im6.set_xdata((2 * np.pi - traj.crun.thetaModulation[i]))
            return [im1, im2, im3, im4, im5, im6, im7]

        def update_time():
            t = -1
            t_max = int(traj.crun.cube.shape[0]) - 1
            while t < t_max:
                #t += anim.direction
                t += 1
                yield t

        anim = animation.FuncAnimation(fig, funcAnim, frames=update_time, interval=1000, blit=False, repeat_delay=1000)
        modulation, mask_cycles, mask_angle, mask_amplitude
        filename = f"results/pyr_mod{modulation}_cycles{mask_cycles}_angle{mask_angle}_amp{int(mask_amplitude*(1e9))}nm_D20_2.mp4"
        writervideo='imagemagick'
        writervideo = animation.FFMpegWriter(fps=1)
        anim.save(filename, writer=writervideo)
        filename = f"results/pyr_mod{modulation}_cycles{mask_cycles}_angle{mask_angle}_amp{int(mask_amplitude * (1e9))}nm_D20_2.gif"
        anim.save(filename, writer='imagemagick', fps=1)


    def viewAndCompareToFITS(self, traj, modulation, mask_cycles, mask_angle, mask_amplitude, fitsFileLocation):
        with fits.open(fitsFileLocation) as hdul:

            hdul.info()
            fitsdata = hdul[0].data

        fitsframes = fitsdata.shape[2]
        fitsdata = np.transpose(fitsdata, (2,0,1))
        filter_function = lambda modulation_l, mask_cycles_l, mask_angle_l, mask_amplitude_l: modulation_l == modulation and \
                                                                                              mask_cycles_l == mask_cycles and \
                                                                                              mask_angle_l == mask_angle and \
                                                                                              mask_amplitude_l == mask_amplitude
        idx_iterator = traj.f_find_idx(['parameters.modulation', 'parameters.mask_cycles', 'parameters.mask_angle', 'parameters.mask_amplitude'], filter_function)
        for idx in idx_iterator:
            traj.v_idx = idx

        fig = plt.figure(figsize=(15, 10))
        #anim_running = True

        ax10 = plt.subplot(4,3,1)
        ax10.set_title("Phase mask (nm)")

        phasemask_view = ax10.imshow(traj.crun.phaseMask)
        plt.colorbar(phasemask_view)

        ax11 = plt.subplot(4,3,2)
        ax11.set_title("Telescope PSF")
        telescope_view = ax11.imshow(np.log10(np.abs(traj.crun.telescope_PSF)), extent=[traj.crun.telescope_xPSF_arcsec[0], traj.crun.telescope_xPSF_arcsec[1], traj.crun.telescope_xPSF_arcsec[0], traj.crun.telescope_xPSF_arcsec[1]])
        telescope_view.set_clim(1, 5)
        ax11.set_xlabel('[Arcsec]')
        ax11.set_ylabel('[Arcsec]')
        plt.colorbar(telescope_view)

        ax12 = plt.subplot(4, 3, 3)
        run_information =  f"Modulation={modulation} lambda/D \n" \
                           f"D = {traj.diameter} m\n" \
                           f"wavelength = {traj.crun.wavelength*(1e9):.2f} nm\n" \
                           f"cycles = {mask_cycles} \n" \
                           f"angle={180*mask_angle/np.pi:.2f} deg\n" \
                           f"mask_amp ={int(mask_amplitude*(1e9))} nm \n" \
                           f"mask_amp meas. (max-min) ={int((np.max(traj.crun.phaseMask)-np.min(traj.crun.phaseMask))*(1e9))} nm \n" \
                           f"mask_amp avg. ={np.mean(traj.crun.phaseMask) * (1e9)} nm \n" \
                           f"sqrt(mean(phaseMask²)) (RMS) = {int(np.sqrt(np.mean(traj.crun.phaseMask ** 2)) * (1e9))} nm \n" \
                           f"phaseMask Pupil only RMS = {int(np.sqrt(np.sum((traj.crun.phaseMask ** 2)/np.sum(traj.crun.telPupil))) * (1e9))} nm \n" \
                           f"magnitude = {traj.magnitude}"
        ax12.set_axis_off()
        ax12.text(0.1, 0.85, run_information,horizontalalignment='left', verticalalignment='center')


        ax1 = plt.subplot(4, 3, 4)
        ax1.set_title("Aberrated\npyramid view")
        ax1_d = plt.subplot(4, 3, 5)
        ax1_d.set_title("Difference\nAber. - ref.")
        ax1_r = plt.subplot(4, 3, 6)
        ax1_r.set_title("Reference\npyramid view")
        ax2 = plt.subplot(4, 3, 7, projection='polar')
        ax2.set_title(r'Beam position and $\Delta \mathrm{I} ( \phi )$')
        ax3 = plt.subplot(4, 3, 8)
        ax3.set_title(
            '$\Delta \mathrm{I} ( \phi )$ as a function\nof the frame number\n(max = ' + str(traj.crun.cube.shape[0]) + ')')
        ax4 = plt.subplot(4, 3, 9)
        ax4.set_title("PSF position relative\nto the pyramid peak")
        axfits = plt.subplot(4, 3, 10)
        axfits.set_title("Experimental data")

        # set the spacing between subplots
        plt.subplots_adjust(left=0.1,
                            bottom=0.1,
                            right=0.9,
                            top=0.9,
                            wspace=0.4,
                            hspace=0.4)

        cube_diff = traj.crun.cube - traj.crun.cube_ref
        distance = (traj.crun.wavelength / traj['diameter']) * traj['modulation']

        ax3.bar(list(range(traj.crun.cube.shape[0])), traj.crun.delta_I)
        ax2.bar(2 * np.pi - traj.crun.thetaModulation, traj.crun.delta_I, width=(2 * np.pi / traj.crun.cube.shape[0]), edgecolor="c")

        im1 = ax1.imshow(traj.crun.cube[0, :, :], cmap=cm.Greys_r)
        im2 = ax1_r.imshow(traj.crun.cube_ref[0, :, :], cmap=cm.Greys_r)
        im3 = ax1_d.imshow(cube_diff[0, :, :], cmap=cm.Greys_r)
        imfits = axfits.imshow(fitsdata[0,:,:], cmap=cm.Greys_r)

        im4 = ax2.scatter((2 * np.pi - traj.crun.thetaModulation[0]), distance, s=200)
        im5 = ax3.axvline(0, color='r')
        ax2.axvline(0, lw=2, alpha=0.5, color='k')
        ax2.axvline(np.pi * 0.5, lw=2, alpha=0.5, color='k')
        ax2.axvline(np.pi, lw=2, alpha=0.5, color='k')
        ax2.axvline(np.pi * 1.5, lw=2, alpha=0.5, color='k')
        im6 = ax2.axvline(0, lw=3, color='r')
        im7 = ax4.imshow(traj.crun.image_cube[0, :, :], cmap=cm.Greys_r)
        mid = traj.crun.image_cube.shape[1] / 2.0
        ax4.axvline(mid, lw=2, alpha=0.5, color='r')
        ax4.axhline(mid, lw=2, alpha=0.5, color='r')

        im1_cbar = plt.colorbar(im1, ax=ax1)
        im1_cbar.ax.set_autoscale_on(True)
        im2_cbar = plt.colorbar(im2, ax=ax1_r)
        im2_cbar.ax.set_autoscale_on(True)
        im3_cbar = plt.colorbar(im3, ax=ax1_d)
        im3_cbar.ax.set_autoscale_on(True)
        imfits_cbar = plt.colorbar(imfits, ax=axfits)
        imfits_cbar.ax.set_autoscale_on(True)

        modulationFrames = traj.crun.cube.shape[0]


        def funcAnim(i):
            frames_to_use = [i*4, (i*4)+1, (i*4)+2, (i*4)+3]
            """ Aberrated wavefront """
            frame = np.sum(traj.crun.cube[frames_to_use, :, :], axis=0)
            im1.set_data(frame)
            im1.set_clim(vmin=frame.min(), vmax=frame.max())
            """ Reference wavefront """
            frame_ref = np.sum(traj.crun.cube_ref[frames_to_use, :, :], axis=0)
            im2.set_data(frame_ref)
            im2.set_clim(vmin=frame_ref.min(), vmax=frame_ref.max())
            """ Difference of aberrated with reference wavefront """
            frame_diff = np.sum(cube_diff[frames_to_use, :, :], axis=0)
            im3.set_data(frame_diff)
            im3.set_clim(vmin=frame_diff.min(), vmax=frame_diff.max())
            """ Experimental data """
            # rightFrame = int(((67-i))*(50/100))
            #rightFrame = i
            imfits.set_data(fitsdata[550-i, :, :])
            imfits.set_clim(vmin=fitsdata[550-i, :, :].min(), vmax=fitsdata[550-i, :, :].max())

            """ Beam position """
            frame_image = np.sum(traj.crun.image_cube[frames_to_use, :, :], axis=0)
            im7.set_data(frame_image)
            im7.set_clim(vmin=frame_image.min(), vmax=frame_image.max())

            ax4.set_xlim(mid - 50, mid + 50)
            ax4.set_ylim(mid - 50, mid + 50)
            im4.set_offsets([(2 * np.pi - traj.crun.thetaModulation[i*2]), distance])
            im5.set_xdata(list(range(len(traj.crun.thetaModulation)))[i*2])
            im6.set_xdata((2 * np.pi - traj.crun.thetaModulation[i*2]))
            return [im1, im2, im3, im4, im5, im6, im7]

        def update_time():
            t = -1
            t_max = int(traj.crun.cube.shape[0]/4) - 1
            while t < t_max:
                #t += anim.direction
                t += 1
                yield t

        anim = animation.FuncAnimation(fig, funcAnim, frames=update_time, interval=1000, blit=False, repeat_delay=1000)
        # modulation, mask_cycles, mask_angle, mask_amplitude
        filename = f"results/pyr_mod{modulation}_cycles{mask_cycles}_angle{mask_angle}_amp{int(mask_amplitude*(1e9))}nm_D8.mp4"
        writervideo='imagemagick'
        writervideo = animation.FFMpegWriter(fps=1)
        anim.save(filename, writer=writervideo)
        filename = f"results/pyr_mod{modulation}_cycles{mask_cycles}_angle{mask_angle}_amp{int(mask_amplitude * (1e9))}nm_D8.gif"
        anim.save(filename, writer='imagemagick', fps=1)

    def viewAndCompareToFITS_reduced(self, traj, modulation, mask_cycles, mask_angle, mask_amplitude, fitsFileLocation, refFitsFileLocation, offset=550, ref_offset=550):
        with fits.open(fitsFileLocation) as hdul:

            hdul.info()
            fitsdata = hdul[0].data

        with fits.open(refFitsFileLocation) as hdul:

            hdul.info()
            REF_fitsdata = hdul[0].data

        #fitsframes = fitsdata.shape[2]
        fitsdata = np.transpose(fitsdata, (2,0,1))
        ref_fitsdata = np.transpose(REF_fitsdata, (2,0,1))

        filter_function = lambda modulation_l, mask_cycles_l, mask_angle_l, mask_amplitude_l: modulation_l == modulation and \
                                                                                              mask_cycles_l == mask_cycles and \
                                                                                              mask_angle_l == mask_angle and \
                                                                                              mask_amplitude_l == mask_amplitude
        idx_iterator = traj.f_find_idx(['parameters.modulation', 'parameters.mask_cycles', 'parameters.mask_angle', 'parameters.mask_amplitude'], filter_function)
        for idx in idx_iterator:
            traj.v_idx = idx

        fig = plt.figure(figsize=(15, 10))
        #anim_running = True

        ax10 = plt.subplot(2,3,1)
        ax10.set_title("Phase mask (nm)")

        phasemask_view = ax10.imshow(traj.crun.phaseMask.T)
        plt.colorbar(phasemask_view)

        # ax11 = plt.subplot(4,3,2)
        # ax11.set_title("Telescope PSF")
        # telescope_view = ax11.imshow(np.log10(np.abs(traj.crun.telescope_PSF)), extent=[traj.crun.telescope_xPSF_arcsec[0], traj.crun.telescope_xPSF_arcsec[1], traj.crun.telescope_xPSF_arcsec[0], traj.crun.telescope_xPSF_arcsec[1]])
        # telescope_view.set_clim(1, 5)
        # ax11.set_xlabel('[Arcsec]')
        # ax11.set_ylabel('[Arcsec]')
        # plt.colorbar(telescope_view)

        # ax12 = plt.subplot(2, 2, 1)
        #run_information = f"Modulation={modulation} lambda/D, D = {traj.diameter} m, wavelength = {traj.crun.wavelength*(1e9):.2f} nm, cycles = {mask_cycles}, angle={180*mask_angle/np.pi:.2f} deg, mask_amp ={int(mask_amplitude*(1e9))} nm, phaseMask Pupil RMS = {int(np.sqrt(np.sum((traj.crun.phaseMask ** 2)/np.sum(traj.crun.telPupil))) * (1e9))} nm, magnitude = {traj.magnitude}"
        #fig.suptitle(run_information, wrap=True)

        ax12 = plt.subplot(2, 3, 6)
        run_information =  f"Modulation={modulation} lambda/D \n" \
                           f"D = {traj.diameter} m\n" \
                           f"wavelength = {traj.crun.wavelength*(1e9):.2f} nm\n" \
                           f"cycles = {mask_cycles} \n" \
                           f"angle={180*mask_angle/np.pi:.2f} deg\n" \
                           f"mask_amp ={int(mask_amplitude*(1e9))} nm \n" \
                           f"phaseMask Pupil only RMS = {int(np.sqrt(np.sum((traj.crun.phaseMask ** 2)/np.sum(traj.crun.telPupil))) * (1e9))} nm \n" \
                           f"magnitude = {traj.magnitude}"
        ax12.set_axis_off()
        ax12.text(0.1, 0.85, run_information,horizontalalignment='left', verticalalignment='center')

        # ax12.set_axis_off()
        # ax12.text(0.1, 0.85, run_information,horizontalalignment='left', verticalalignment='center')


        ax1 = plt.subplot(2, 3, 2)
        ax1.set_title("Simulated\npyramid view")
        # ax1_d = plt.subplot(4, 3, 5)
        # ax1_d.set_title("Difference\nAber. - ref.")
        # ax1_r = plt.subplot(4, 3, 6)
        # ax1_r.set_title("Reference\npyramid view")
        ax2 = plt.subplot(2, 3, 4, projection='polar')
        ax2.set_title(r'Beam position and $\Delta \mathrm{I} ( \phi )$')
        # ax3 = plt.subplot(4, 3, 8)
        # ax3.set_title(
        #     '$\Delta \mathrm{I} ( \phi )$ as a function\nof the frame number\n(max = ' + str(traj.crun.cube.shape[0]) + ')')
        ax4 = plt.subplot(2, 3, 5)
        ax4.set_title("PSF position relative\nto the pyramid peak")
        axfits = plt.subplot(2, 3, 3)
        axfits.set_title("Experimental\npyramid view")

        # set the spacing between subplots
        plt.subplots_adjust(left=0.1,
                            bottom=0.1,
                            right=0.9,
                            top=0.9,
                            wspace=0.4,
                            hspace=0.4)

        cube_diff = traj.crun.cube - traj.crun.cube_ref
        cube_diff = cube_diff + np.abs(np.min(cube_diff))
        distance = (traj.crun.wavelength / traj['diameter']) * traj['modulation']

        # ax3.bar(list(range(traj.crun.cube.shape[0])), traj.crun.delta_I)
        ax2.bar(2 * np.pi - traj.crun.thetaModulation, traj.crun.delta_I, width=(2 * np.pi / traj.crun.cube.shape[0]), edgecolor="c")

        im1 = ax1.imshow(traj.crun.cube[0, :, :], cmap=cm.Greys_r)
        # im2 = ax1_r.imshow(traj.crun.cube_ref[0, :, :], cmap=cm.Greys_r)
        # im3 = ax1_d.imshow(cube_diff[0, :, :], cmap=cm.Greys_r)
        imfits = axfits.imshow(fitsdata[0,:,:]-ref_fitsdata[0,:,:], cmap=cm.Greys_r)

        #im4 = ax2.scatter((traj.crun.thetaModulation[0]), distance, s=200)
        # im5 = ax3.axvline(0, color='r')
        ax2.axvline(0, lw=2, alpha=0.5, color='k')
        ax2.axvline(np.pi * 0.5, lw=2, alpha=0.5, color='k')
        ax2.axvline(np.pi, lw=2, alpha=0.5, color='k')
        ax2.axvline(np.pi * 1.5, lw=2, alpha=0.5, color='k')
        im6 = ax2.axvline(np.pi, lw=3, color='r')
        im7 = ax4.imshow( np.log10(np.abs(traj.crun.image_cube[0, :, :])) , cmap=cm.Greys_r)

        mid = traj.crun.image_cube.shape[1] / 2.0
        ax4.axvline(mid, lw=2, alpha=0.5, color='r')
        ax4.axhline(mid, lw=2, alpha=0.5, color='r')

        im1_cbar = plt.colorbar(im1, ax=ax1)
        im1_cbar.ax.set_autoscale_on(True)
        # im2_cbar = plt.colorbar(im2, ax=ax1_r)
        # im2_cbar.ax.set_autoscale_on(True)
        # im3_cbar = plt.colorbar(im3, ax=ax1_d)
        # im3_cbar.ax.set_autoscale_on(True)
        imfits_cbar = plt.colorbar(imfits, ax=axfits)
        imfits_cbar.ax.set_autoscale_on(True)

        modulationFrames = traj.crun.cube.shape[0]


        def funcAnim(i):
            frames_to_use = [i*4, (i*4)+1, (i*4)+2, (i*4)+3]
            """ Difference of aberrated with reference wavefront wavefront """
            frame = np.sum(cube_diff[frames_to_use, :, :], axis=0)
            # frame = np.sum(traj.crun.cube[frames_to_use, :, :], axis=0)
            # frame = frame + np.min(frame)
            im1.set_data(frame)
            im1.set_clim(vmin=frame.min(), vmax=frame.max())
            """ Reference wavefront """
            frame_ref = np.sum(traj.crun.cube_ref[frames_to_use, :, :], axis=0)
            # im2.set_data(frame_ref)
            # im2.set_clim(vmin=frame_ref.min(), vmax=frame_ref.max())
            """ Difference of aberrated with reference wavefront """
            #
            # im3.set_data(frame_diff)
            # im3.set_clim(vmin=frame_diff.min(), vmax=frame_diff.max())
            """ Experimental data """
            # rightFrame = int(((67-i))*(50/100))
            #rightFrame = i
            imfits.set_data(fitsdata[offset-i, :, :]-ref_fitsdata[ref_offset-i, :, :])
            imfits.set_clim(vmin=(fitsdata[offset-i, :, :]-ref_fitsdata[ref_offset-i, :, :]).min(), vmax=(fitsdata[offset-i, :, :]-ref_fitsdata[ref_offset-i, :, :]).max())

            """ Beam position """
            #frames_to_use_for_image = (traj.crun.image_cube.shape[0]) + np.array(frames_to_use)
            frames_to_use_for_image = np.array(frames_to_use)
            frame_image = np.sum(traj.crun.image_cube[frames_to_use_for_image, :, :], axis=0)

            im7.set_data(np.log10(frame_image))
            im7.set_clim([3.7,5])
            # im7.set_data(frame_image)
            # im7.set_clim(vmin=frame_image.min(), vmax=frame_image.max())

            ax4.set_xlim(mid + 40, mid - 40)
            ax4.set_ylim(mid - 40, mid + 40)
            #im4.set_offsets([(traj.crun.thetaModulation[i*4]), distance])
            # im5.set_xdata(list(range(len(traj.crun.thetaModulation)))[i*2])
            im6.set_xdata(np.pi - (np.mean(traj.crun.thetaModulation[frames_to_use])))
            return [im1, im6, im7]

        def update_time():
            t = -1
            t_max = int(traj.crun.cube.shape[0]/4) - 1
            while t < t_max:
                #t += anim.direction
                t += 1
                yield t

        anim = animation.FuncAnimation(fig, funcAnim, frames=update_time, interval=1000, blit=False, repeat_delay=1000)
        # modulation, mask_cycles, mask_angle, mask_amplitude
        filename = f"/mnt/home/usager/cars2019/Documents/Programming/OOPAO_TRWFS/trwfs/vis/results/pyr_mod{modulation}_cycles{mask_cycles}_angle{mask_angle}_amp{int(mask_amplitude*(1e9))}nm_D8.mp4"
        writervideo='imagemagick'
        writervideo = animation.FFMpegWriter(fps=1)
        anim.save(filename, writer=writervideo)
        filename = f"/mnt/home/usager/cars2019/Documents/Programming/OOPAO_TRWFS/trwfs/vis/results/pyr_mod{modulation}_cycles{mask_cycles}_angle{mask_angle}_amp{int(mask_amplitude * (1e9))}nm_D8.gif"
        anim.save(filename, writer='imagemagick', fps=1)


def dict_to_trajectory(d, traj):
    for k,v in d.items():
        traj.f_add_parameter(k, v, comment=k)


if __name__ == "__main__":
    from trwfs.parameter_files.parameterFile_CMOS_PWFS_aug2022_3 import initializeParameterFile

    PPS = PyramidPropagationSimulation()
    param = initializeParameterFile()

    # Create an environment that handles running our simulation
    env = Environment(trajectory='run', filename='/home/cars2019/DATA/HDF/florence_data_batch_final_alt_high_res.hdf5',
                      file_title='trpwfs_image_simulation_D8',
                      comment='Simulation to recreate the experimental results of Florence',
                      large_overview_tables=True,
                      log_config='DEFAULT',
                      log_stdout=True,
                      overwrite_file=True)

    # Get the trajectory from the environment
    traj = env.trajectory

    dict_to_trajectory(param, traj)
    traj.f_add_parameter('mask_cycles', 1.0, comment='Cycles of the Fourier mask')
    traj.f_add_parameter('mask_angle', 0.0, comment='Angle of the Fourier mask (in rad)')
    traj.f_add_parameter('mask_amplitude', 600*(1e-9), comment='Amplitude of the mask (in m). Corresponds to the RMS of the distortion induced.')

    traj.f_explore(cartesian_product({'modulation': [3, 5],
                                      'mask_cycles': [4.0, 7.9],
                                      'mask_amplitude': [0.0, 15*(1e-9), 30*(1e-9)],
                                      'nTheta_user_defined': [100],
                                      'mask_angle': [np.pi/4]}))


    env.run(PPS.run)

    # Let's check that all runs are completed!
    assert traj.f_is_completed()

    # Finally disable logging and close all log-files
    env.disable_logging()

