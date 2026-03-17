from OOPAO.Pyramid import *
from trwfs.tools.PWFS_CMOS_NOISE_PROOF_may2022_tools import *


class TR_Pyramid(Pyramid):
    """
    Time-Resolved Pyramid wavefront sensor. Inherits the Pyramid class and changes a few lines in order to keeps all the
    frames for each angle of the modulation. Where you see "cube", each frame is the for a different angle of the
    modulation.
    """

    def __init__(self, **kwargs):
        """ We can directly define the removed frames for each image we will want to generate"""
        self.nTheta_user_defined = kwargs["nTheta_user_defined"]
        if self.nTheta_user_defined:
            if (((self.nTheta_user_defined // 4) % 2) == 0):
                self.remFrames = np.array([0])
                frms = np.array(list(range((self.nTheta_user_defined // 4))))
                self.remFrames = np.append(self.remFrames, frms[frms % 2 == 1])  # [0,1,3,5,7,9,11]
            else:
                frms = np.array(list(range((self.nTheta_user_defined // 4))))
                self.remFrames = frms[frms % 2 == 0]

        self.referenceSignalCube_2D = None
        self.referenceSignalCube = None
        super().__init__(**kwargs)


    def wfs_calibration(self, telescope):
        # reference slopes acquisition
        telescope.OPD = telescope.pupil.astype(float)
        # compute the refrence slopes
        self.wfs_measure(phase_in=self.telescope.src.phase)
        if (self.modulation == 0):
            self.referenceSignal_2D, self.referenceSignal = self.signalProcessing()
        else:
            self.referenceSignal_2D, self.referenceSignal, self.referenceSignalCube_2D, self.referenceSignalCube = self.signalProcessing()

        # 2D reference Frame before binning with detector
        self.referencePyramidFrame = np.copy(self.pyramidFrame)
        self.referencePyramidCube  = np.copy(self.pyramidCube)
        if self.isCalibrated is False:
            print('WFS calibrated!')
        self.isCalibrated = True
        telescope.OPD = telescope.pupil.astype(float)

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% PYRAMID PROPAGATION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    def pyramid_propagation(self, telescope):
        # backward compatibility with previous version
        self.wfs_measure(phase_in=telescope.src.phase)
        return

    def wfs_measure(self, phase_in=None):
        if phase_in is not None:
            self.telescope.src.phase = phase_in
        # mask amplitude for the light propagation
        self.maskAmplitude = self.convert_for_gpu(self.telescope.pupilReflectivity)

        if self.spatialFilter is not None:
            if np.ndim(phase_in) == 2:
                support_spatial_filter = np.copy(self.supportPadded)
                em_field = self.maskAmplitude * np.exp(1j * (self.telescope.src.phase))
                support_spatial_filter[
                self.center - self.telescope.resolution // 2:self.center + self.telescope.resolution // 2,
                self.center - self.telescope.resolution // 2:self.center + self.telescope.resolution // 2] = em_field
                self.em_field_spatial_filter = (np.fft.fft2(support_spatial_filter * self.phasor))
                self.pupil_plane_spatial_filter = (np.fft.ifft2(self.em_field_spatial_filter * self.spatialFilter))

        # modulation camera
        self.modulation_camera_em = []
        self.modulation_camera_em_phase = []

        if self.modulation == 0:
            if np.ndim(phase_in) == 2:
                self.pyramidFrame = self.convert_for_numpy(
                    self.pyramid_transform(self.convert_for_gpu(self.telescope.src.phase)))
                self * self.cam
                if self.isInitialized and self.isCalibrated:
                    self.pyramidSignal_2D, self.pyramidSignal = self.signalProcessing()
            else:
                nModes = phase_in.shape[2]
                # move axis to get the number of modes first
                self.phase_buffer = self.convert_for_gpu(np.moveaxis(self.telescope.src.phase, -1, 0))

                # define the parallel jobs
                def job_loop_multiple_modes_non_modulated():
                    Q = Parallel(n_jobs=self.nJobs, prefer=self.joblib_setting)(
                        delayed(self.pyramid_transform)(i) for i in self.phase_buffer)
                    return Q
                    # apply the pyramid transform in parallel

                maps = job_loop_multiple_modes_non_modulated()

                self.pyramidSignal_2D = np.zeros([self.validSignal.shape[0], self.validSignal.shape[1], nModes])
                self.pyramidSignal = np.zeros([self.nSignal, nModes])
                self.pyramidSignalCube = np.zeros([len(self.remFrames), self.nSignal, nModes])
                self.pyramidSignalCube_2D = np.zeros([len(self.remFrames), self.validSignal.shape[0], self.validSignal.shape[1], nModes])

                for i in range(nModes):
                    self.pyramidFrame = self.convert_for_numpy(maps[i])
                    self * self.cam
                    if self.isInitialized:
                        self.pyramidSignal_2D[:, :, i], self.pyramidSignal[:, i] = self.signalProcessing()
                del maps

        else:
            if np.ndim(phase_in) == 2:
                n_max_ = self.n_max
                if self.nTheta > n_max_:
                    # break problem in pieces:
                    print("too many ntheta")
                    nCycle = int(np.ceil(self.nTheta / n_max_))
                    # print(self.nTheta)
                    maps = self.convert_for_numpy(np_cp.zeros([self.nRes, self.nRes]))
                    for i in range(nCycle):
                        if self.gpu_available:
                            try:
                                self.mempool = np_cp.get_default_memory_pool()
                                self.mempool.free_all_blocks()
                            except:
                                print('could not free the memory')
                        if i < nCycle - 1:
                            def job_loop_single_mode_modulated():
                                Q = Parallel(n_jobs=self.nJobs, prefer=self.joblib_setting)(
                                    delayed(self.pyramid_transform)(i) for i in self.convert_for_gpu(
                                        self.phaseBuffModulationLowres[i * n_max_:(i + 1) * n_max_, :, :]))
                                return Q

                            maps += self.convert_for_numpy(
                                np_cp.sum(np_cp.asarray(job_loop_single_mode_modulated()), axis=0))
                        else:
                            def job_loop_single_mode_modulated():
                                Q = Parallel(n_jobs=self.nJobs, prefer=self.joblib_setting)(
                                    delayed(self.pyramid_transform)(i) for i in
                                    self.convert_for_gpu(self.phaseBuffModulationLowres[i * n_max_:, :, :]))
                                return Q

                            maps += self.convert_for_numpy(
                                np_cp.sum(np_cp.asarray(job_loop_single_mode_modulated()), axis=0))
                    self.pyramidFrame = maps / self.nTheta
                    del maps
                else:
                    #print("Right number of ntheta")
                    # define the parallel jobs
                    def job_loop_single_mode_modulated():
                        Q = Parallel(n_jobs=self.nJobs, prefer=self.joblib_setting)(
                            delayed(self.pyramid_transform)(i, p) for i,p in zip(self.phaseBuffModulationLowres, self.thetaModulation))
                        return Q
                        # apply the pyramid transform in parallel

                    self.maps = np_cp.asarray(job_loop_single_mode_modulated())
                    # compute the sum of the pyramid frames for each modulation points
                    if self.weight_vector is None:
                        self.pyramidFrame = self.convert_for_numpy(np_cp.sum((self.maps), axis=0)) / self.nTheta
                        self.pyramidCube = self.convert_for_numpy(np_cp.asarray(self.maps)) / self.nTheta
                    else:
                        weighted_map = np.reshape(self.maps, [self.nTheta, self.nRes ** 2])
                        self.weighted_map = np.diag(self.weight_vector) @ weighted_map
                        self.pyramidFrame = np.reshape(
                            self.convert_for_numpy(np_cp.sum((self.weighted_map), axis=0)) / self.nTheta,
                            [self.nRes, self.nRes])

                        self.pyramidCube = np.reshape(
                            self.convert_for_numpy(np_cp.asarray(self.weighted_map)) / self.nTheta,
                            [self.nRes, self.nRes])

                # propagate to the detector
                self * self.cam

                if self.isInitialized and self.isCalibrated:
                    self.pyramidSignal_2D, self.pyramidSignal, self.pyramidSignalCube_2D, self.pyramidSignalCube = self.signalProcessing()
            else:
                if np.ndim(phase_in) == 3:
                    nModes = phase_in.shape[2]
                    # move axis to get the number of modes first
                    self.phase_buffer = np.moveaxis(self.telescope.src.phase, -1, 0)

                    def jobLoop_setPhaseBuffer():
                        Q = Parallel(n_jobs=self.nJobs, prefer=self.joblib_setting)(
                            delayed(self.setPhaseBuffer)(i) for i in self.phase_buffer)
                        return Q

                    self.phaseBuffer = (np.reshape(np.asarray(jobLoop_setPhaseBuffer()),
                                                   [nModes * self.nTheta, self.telescope.resolution,
                                                    self.telescope.resolution]))
                    n_measurements = nModes * self.nTheta
                    n_max = self.n_max
                    n_measurement_max = int(np.floor(n_max / self.nTheta))
                    maps = np_cp.zeros([n_measurements, self.nRes, self.nRes])

                    if n_measurements > n_max:
                        nCycle = int(np.ceil(nModes / n_measurement_max))
                        for i in range(nCycle):
                            if self.gpu_available:
                                try:
                                    self.mempool = np_cp.get_default_memory_pool()
                                    self.mempool.free_all_blocks()
                                except:
                                    print('could not free the memory')
                            if i < nCycle - 1:
                                def job_loop_multiple_mode_modulated():
                                    Q = Parallel(n_jobs=self.nJobs, prefer=self.joblib_setting)(
                                        delayed(self.pyramid_transform)(i) for i in self.convert_for_gpu(
                                            self.phaseBuffer[i * n_measurement_max * self.nTheta:(
                                                                                                             i + 1) * n_measurement_max * self.nTheta,
                                            :, :]))
                                    return Q

                                maps[i * n_measurement_max * self.nTheta:(i + 1) * n_measurement_max * self.nTheta, :,
                                :] = np_cp.asarray(job_loop_multiple_mode_modulated())
                            else:
                                def job_loop_multiple_mode_modulated():
                                    Q = Parallel(n_jobs=self.nJobs, prefer=self.joblib_setting)(
                                        delayed(self.pyramid_transform)(i) for i in self.convert_for_gpu(
                                            self.phaseBuffer[i * n_measurement_max * self.nTheta:, :, :]))
                                    return Q

                                maps[i * n_measurement_max * self.nTheta:, :, :] = np_cp.asarray(
                                    job_loop_multiple_mode_modulated())
                        self.bufferPyramidFrames = self.convert_for_numpy(maps)
                        del self.phaseBuffer
                        del maps
                        if self.gpu_available:
                            try:
                                self.mempool = np_cp.get_default_memory_pool()
                                self.mempool.free_all_blocks()
                            except:
                                print('could not free the memory')
                    else:
                        def job_loop_multiple_mode_modulated():
                            Q = Parallel(n_jobs=self.nJobs, prefer=self.joblib_setting)(
                                delayed(self.pyramid_transform)(i) for i in self.convert_for_gpu(self.phaseBuffer))
                            return Q

                        self.bufferPyramidFrames = self.convert_for_numpy(
                            np_cp.asarray(job_loop_multiple_mode_modulated()))

                    self.pyramidSignal_2D = np.zeros([self.validSignal.shape[0], self.validSignal.shape[1], nModes])
                    self.pyramidSignal = np.zeros([self.nSignal, nModes])
                    self.pyramidSignalCube = np.zeros([len(self.remFrames), self.nSignal, nModes])
                    self.pyramidSignalCube_2D = np.zeros([len(self.remFrames), self.validSignal.shape[0], self.validSignal.shape[1], nModes])

                    for i in range(nModes):
                        self.pyramidFrame = np_cp.sum(
                            self.bufferPyramidFrames[i * (self.nTheta):(self.nTheta) + i * (self.nTheta)],
                            axis=0) / self.nTheta
                        self.pyramidCube = (self.bufferPyramidFrames[i * (self.nTheta):(self.nTheta) + i * (self.nTheta)]) / self.nTheta
                        self * self.cam
                        if self.isInitialized:
                            self.pyramidSignal_2D[:, :, i], self.pyramidSignal[:, i], self.pyramidSignalCube_2D[:,:,:,i], self.pyramidSignalCube[:,:,i]  = self.signalProcessing()
                    del self.bufferPyramidFrames
                else:
                    print(
                        '%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
                    print('Error - Wrong dimension for the input phase. Aborting....')
                    print('Aborting...')
                    print(
                        '%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
                    sys.exit(0)
                if self.gpu_available:
                    try:
                        self.mempool = np_cp.get_default_memory_pool()
                        self.mempool.free_all_blocks()
                    except:
                        print('could not free the memory')

    def pyramid_transform(self,phase_in, phase=0):
        # copy of the support for the zero-padding
        support = self.supportPadded.copy()
        # em field corresponding to phase_in
        if np.ndim(self.telescope.OPD)==2:
            if self.modulation==0:
                em_field     = self.maskAmplitude*np.exp(1j*(phase_in))
            else:
                em_field     = self.maskAmplitude*np.exp(1j*(self.convert_for_gpu(self.telescope.src.phase)+phase_in))
        else:
            em_field     = self.maskAmplitude*np.exp(1j*phase_in)
        # zero-padding for the FFT computation
        support[self.center-self.telescope.resolution//2:self.center+self.telescope.resolution//2,self.center-self.telescope.resolution//2:self.center+self.telescope.resolution//2] = em_field
        del em_field
        # case with mask centered on 4 pixels
        if self.psfCentering:
            em_field_ft     = np_cp.fft.fft2(support*self.phasor)
            em_field_pwfs   = np_cp.fft.ifft2(em_field_ft*self.mask)
            I               = np_cp.abs(em_field_pwfs)**2
        # case with mask centered on 1 pixel
        else:
            if self.spatialFilter is not None:
                em_field_ft     = np_cp.fft.fftshift(np_cp.fft.fft2(support))*self.spatialFilter
            else:
                em_field_ft     = np_cp.fft.fftshift(np_cp.fft.fft2(support))

            em_field_pwfs   = np_cp.fft.ifft2(em_field_ft*self.mask)
            I               = np_cp.abs(em_field_pwfs)**2
        del support
        del em_field_pwfs
        self.modulation_camera_em.append(self.convert_for_numpy(em_field_ft))
        self.modulation_camera_em_phase.append(phase)

        del em_field_ft
        del phase_in



        return I

    def pyramid_transform_single_frame(self, frame_number, max_frames):

        dTheta = frame_number * ((2*np.pi)/max_frames)
        TT = (self.modulation * (np.cos(dTheta) * self.Tip + np.sin(dTheta) * self.Tilt)) * self.telescope.pupil

        phase_in = TT

        # copy of the support for the zero-padding
        support = self.supportPadded.copy()
        # em field corresponding to phase_in

        if np.ndim(self.telescope.OPD) == 2:
            if self.modulation == 0:
                em_field = self.maskAmplitude * np.exp(1j * (phase_in))
            else:
                em_field = self.maskAmplitude * np.exp(1j * (self.convert_for_gpu(self.telescope.src.phase) + phase_in))
        else:
            em_field = self.maskAmplitude * np.exp(1j * phase_in)

        # zero-padding for the FFT computation
        support[self.center - self.telescope.resolution // 2:self.center + self.telescope.resolution // 2,
        self.center - self.telescope.resolution // 2:self.center + self.telescope.resolution // 2] = em_field

        # case with mask centered on 4 pixels
        if self.psfCentering:
            em_field_ft = np_cp.fft.fft2(support * self.phasor)
            em_field_pwfs = np_cp.fft.ifft2(em_field_ft * self.mask)
            I = np_cp.abs(em_field_pwfs) ** 2

        # case with mask centered on 1 pixel
        else:
            em_field_ft = np_cp.fft.fftshift(np_cp.fft.fft2(support))
            em_field_pwfs = np_cp.fft.ifft2(em_field_ft * self.mask)
            I = np_cp.abs(em_field_pwfs) ** 2

        return I

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% WFS SIGNAL PROCESSING %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    def saveGiftest(self, filename, data, framerate=1):
        fname, _ = os.path.splitext(filename)  # split the extension by last period
        filename = fname + '.gif'  # ensure the .gif extension
        if data.ndim == 3:  # If number of dimensions are 3,
            data = data[..., np.newaxis] * np.ones(
                3)  # copy into the color  dimension if images are black and white
        clip = ImageSequenceClip(list(data), fps=framerate)
        clip.write_gif(filename, fps=framerate)

    def signalProcessing(self, cameraFrame=None):
        if cameraFrame is None:
            cameraFrame = self.cam.frame
            cameraCube = self.cam.cube

        if self.postProcessing == 'slopesMaps':
            # slopes-maps computation
            I1 = self.grabQuadrant(1, cameraFrame=0) * self.validI4Q
            I2 = self.grabQuadrant(2, cameraFrame=0) * self.validI4Q
            I3 = self.grabQuadrant(3, cameraFrame=0) * self.validI4Q
            I4 = self.grabQuadrant(4, cameraFrame=0) * self.validI4Q
            # global normalisation
            I4Q = I1 + I2 + I3 + I4
            norma = np.mean(I4Q[self.validI4Q])
            # slopesMaps computation cropped to the valid pixels
            Sx = (I1 - I2 + I4 - I3)
            Sy = (I1 - I4 + I2 - I3)
            # 2D slopes maps
            slopesMaps = (np.concatenate((Sx, Sy) / norma) - self.referenceSignal_2D) * self.slopesUnits
            # slopes vector
            slopes = slopesMaps[np.where(self.validSignal == 1)]
            return slopesMaps, slopes

        if self.postProcessing == 'slopesMaps_incidence_flux':
            # slopes-maps computation
            I1 = self.grabQuadrant(1, cameraFrame=0) * self.validI4Q
            I2 = self.grabQuadrant(2, cameraFrame=0) * self.validI4Q
            I3 = self.grabQuadrant(3, cameraFrame=0) * self.validI4Q
            I4 = self.grabQuadrant(4, cameraFrame=0) * self.validI4Q

            # global normalisation
            I4Q = I1 + I2 + I3 + I4
            subArea = (self.telescope.D / self.nSubap) ** 2
            norma = np.float64(self.telescope.src.nPhoton * self.telescope.samplingTime * subArea)

            # slopesMaps computation cropped to the valid pixels
            Sx = (I1 - I2 + I4 - I3)
            Sy = (I1 - I4 + I2 - I3)

            # 2D slopes maps
            slopesMaps = (np.concatenate((Sx, Sy) / norma) - self.referenceSignal_2D) * self.slopesUnits

            # slopes vector
            slopes = slopesMaps[np.where(self.validSignal == 1)]
            return slopesMaps, slopes

        if self.postProcessing == 'fullFrame_incidence_flux':
            # global normalization
            subArea = (self.telescope.D / self.nSubap) ** 2
            norma = np.float64(self.telescope.src.nPhoton * self.telescope.samplingTime * subArea) / 4
            # 2D full-frame
            fullFrameMaps = (cameraFrame / norma) - self.referenceSignal_2D
            # full-frame vector
            fullFrame = fullFrameMaps[np.where(self.validSignal == 1)]

            return fullFrameMaps, fullFrame
        if (self.postProcessing == 'fullFrame') and (self.modulation == 0):
            # global normalization
            norma = np.sum(cameraFrame[self.validSignal])
            # 2D full-frame
            fullFrameMaps = (cameraFrame / norma) - self.referenceSignal_2D
            # full-frame vector
            fullFrame = fullFrameMaps[np.where(self.validSignal == 1)]

            return fullFrameMaps, fullFrame
        if self.postProcessing == 'fullFrame':
            # global normalization
            norma = np.sum(cameraFrame[self.validSignal])
            # 2D full-frame
            fullFrameMaps = (cameraFrame / norma) - self.referenceSignal_2D
            # full-frame vector
            fullFrame = fullFrameMaps[np.where(self.validSignal == 1)]
            # With removed selected Frames
            fullFrameMaps_removed = np.zeros((len(self.remFrames), fullFrameMaps.shape[0], fullFrameMaps.shape[1]))
            fullFrame_removed = np.zeros((len(self.remFrames), np.sum(self.validSignal)))
            # self.saveGiftest('test_gif_no_remove.gif', cameraCube/np.max(cameraCube)*255)
            for i in range(len(self.remFrames)):
                removeFramesIdx = calcEquidistantFrameIndices(int(self.remFrames[i]), self.nTheta)
                t_cube = np.delete(cameraCube, removeFramesIdx, axis=0)
                # self.saveGiftest(f'test_gif_remove_{i}.gif', t_cube/np.max(cameraCube)*255)
                if self.referenceSignalCube_2D is None:
                    n_photons = np.sum(t_cube[:,self.validSignal])
                    fullFrameMaps_removed[i, :, :] = (np.sum(t_cube, axis=0) / n_photons)
                else:
                    n_photons = np.sum(t_cube[:,self.validSignal])
                    fullFrameMaps_removed[i, :, :] = (np.sum(t_cube, axis=0) / n_photons) - (
                    self.referenceSignalCube_2D[i, :, :])

                t_frame_rev = fullFrameMaps_removed[i, :, :]
                fullFrame_removed[i, :] = t_frame_rev[np.where(self.validSignal == 1)]

            return fullFrameMaps, fullFrame, fullFrameMaps_removed, fullFrame_removed



    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% WFS PROPERTIES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    @property
    def pyramidSignalCube(self):
        return self._pyramidSignalCube

    @pyramidSignalCube.setter
    def pyramidSignalCube(self,val):
        self._pyramidSignalCube = val
        self.signalCube = val

    @property
    def pyramidSignalCube_2D(self):
        return self._pyramidSignalCube_2D

    @pyramidSignalCube_2D.setter
    def pyramidSignalCube_2D(self, val):
        self._pyramidSignalCube_2D = val
        self.signalCube_2D = val

    def __mul__(self, obj):
        if obj.tag == 'detector':
            I = self.pyramidFrame
            I_t = self.pyramidCube
            obj.frame = (obj.rebin(I, (obj.resolution, obj.resolution)))
            obj.cube = np.zeros([I_t.shape[0], obj.resolution,obj.resolution])
            tmp = np.sum(np.sum(I_t))
            tmp2 = np.sum(I)
            for i in range(I_t.shape[0]):
                obj.cube[i,:,:] = (obj.rebin(I_t[i],(obj.resolution,obj.resolution)))

            if self.binning != 1:
                try:
                    obj.frame = (
                        obj.rebin(obj.frame, (obj.resolution // self.binning, obj.resolution // self.binning)))
                except:
                    print('ERROR: the shape of the detector (' + str(
                        obj.frame.shape) + ') is not valid with the binning value requested:' + str(
                        self.binning) + '!')
            obj.frame = obj.frame * (self.telescope.src.fluxMap.sum()) / obj.frame.sum()
            tmp3 = obj.cube.sum(axis=(1, 2))
            cub_alt = obj.cube * (self.telescope.src.fluxMap.sum() / self.nTheta) / tmp3[:, np.newaxis, np.newaxis]
            obj.cube = obj.cube * (self.telescope.src.fluxMap.sum()) / obj.cube.sum()

            tmp_cube_sum = np.sum(obj.cube, axis=0)

            ''' Unsure how noise should be applied so that the cube and frame "suffer" the same noise equally. 
            For some reason, applying noise to the frame and cube seperately causes an unequal number of photons for 
            each case when you expect the poisson noise to add the same number of photons'''
            if obj.photonNoise != 0:
                for i in range(obj.cube.shape[0]):
                    obj.cube[i, :, :] = self.random_state_photon_noise.poisson(obj.cube[i, :, :])
                ''' For now, I'm applying the noise to the cube, then sum it to create the noisy frame image'''
                obj.cube = np.int64(obj.cube)
                obj.frame = np.sum(obj.cube, axis=0)


            if obj.readoutNoise != 0:
                obj.frame += np.int64(np.round(
                    self.random_state_readout_noise.randn(obj.resolution, obj.resolution) * obj.readoutNoise))
            #                obj.frame = np.round(obj.frame)

            if self.backgroundNoise is True:
                self.backgroundNoiseAdded = self.random_state_background.poisson(self.backgroundNoiseMap)
                obj.frame += self.backgroundNoiseAdded
        else:
            print('Error light propagated to the wrong type of object')
        return -1


# if obj.photonNoise != 0:
#     # rs=np.random.RandomState(seed=int(time.time()))
#     # obj.frame = rs.poisson(obj.frame)
#     obj.frame = self.random_state_photon_noise.poisson(obj.frame)
#
#     for i in range(obj.cube.shape[0]):
#         obj.cube[i, :, :] = self.random_state_photon_noise.poisson(obj.cube[i, :, :])
#         # obj.cube[i, :, :] = rs.poisson(obj.cube[i, :, :])
