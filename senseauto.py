
import os
import numpy as np
import ismrmrd.xsd
import cv2 as cv
from numpy.fft import fftshift, ifftshift, fftn, ifftn
from ismrmrdtools import show, transform
import matplotlib.image
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
import matplotlib.gridspec as gridspec
from ss_v13 import coil_aliased
from ss_all_in_one_v2 import coil_aliased_aio
from scipy import ndimage
from skimage.filters import threshold_otsu, gaussian
from skimage.morphology import binary_closing
from scipy.interpolate import RegularGridInterpolator
from numpy.polynomial.polynomial import polyfit, polyval
import SimpleITK as sitk
from sewar.full_ref import *
import reconstruction_functions as rf
import csv
from multiprocessing import Pool
import multiprocessing
import warnings



# args is a list of lists. Each list is a patient, with objects: Path for Dataset 1, Path for Dataset 2, Patient index,
# First slice, Last Slice + 1, Offset y, Offset x.
# Last 4 elements are not necessary if it is not planned to crop only the prostate area. Values won't be used and can be
# any value. Those values were chosen visually looking at slices plotted.
args = [[
            '/Users/tiago/Documents/FCT/Tese/files_for_tiago/data/0004/kspaces/meas_MID00460_FID707792_T2_TSE_tra_obl-out_1.mrd',
            '/Users/tiago/Documents/FCT/Tese/files_for_tiago/data/0004/kspaces/meas_MID00460_FID707792_T2_TSE_tra_obl-out_2.mrd',
            4,5,12,5,30],
        [
            '/Users/tiago/Documents/FCT/Tese/files_for_tiago/data/0005/kspaces/meas_MID00427_FID709025_T2_TSE_tra_obl-out_1.mrd',
            '/Users/tiago/Documents/FCT/Tese/files_for_tiago/data/0005/kspaces/meas_MID00427_FID709025_T2_TSE_tra_obl-out_2.mrd',
            5,9,18,0,0], [
            '/Users/tiago/Documents/FCT/Tese/files_for_tiago/data/0006/kspaces/meas_MID00392_FID708990_T2_TSE_tra_obl-out_1.mrd'
            ,
            '/Users/tiago/Documents/FCT/Tese/files_for_tiago/data/0006/kspaces/meas_MID00392_FID708990_T2_TSE_tra_obl-out_2.mrd'
            , 6,7,18,0,5], [
            '/Users/tiago/Documents/FCT/Tese/files_for_tiago/data/0007/kspaces/meas_MID00670_FID710360_T2_TSE_tra_obl-out_1.mrd'
            ,
            '/Users/tiago/Documents/FCT/Tese/files_for_tiago/data/0007/kspaces/meas_MID00670_FID710360_T2_TSE_tra_obl-out_2.mrd'
            , 7,10,16,0,-10], [
            '/Users/tiago/Documents/FCT/Tese/files_for_tiago/data/0008/kspaces/meas_MID00224_FID710582_T2_TSE_tra_obl-out_1.mrd'
            ,
            '/Users/tiago/Documents/FCT/Tese/files_for_tiago/data/0008/kspaces/meas_MID00224_FID710582_T2_TSE_tra_obl-out_2.mrd'
            , 8,14,24,-10,10], [
            '/Users/tiago/Documents/FCT/Tese/files_for_tiago/data/0009/kspaces/meas_MID00701_FID711740_T2_TSE_tra_obl-out_1.mrd'
            ,
            '/Users/tiago/Documents/FCT/Tese/files_for_tiago/data/0009/kspaces/meas_MID00701_FID711740_T2_TSE_tra_obl-out_2.mrd'
            , 9,9,20,0,5], [
            '/Users/tiago/Documents/FCT/Tese/files_for_tiago/data/0010/kspaces/meas_MID00060_FID711794_T2_TSE_tra_obl-out_1.mrd'
            ,
            '/Users/tiago/Documents/FCT/Tese/files_for_tiago/data/0010/kspaces/meas_MID00060_FID711794_T2_TSE_tra_obl-out_2.mrd'
            , 10,8,20,0,0], [
            '/Users/tiago/Documents/FCT/Tese/files_for_tiago/data/0002/kspaces/meas_MID00039_FID701346_T2_TSE_tra_obl-out_1.mrd'
            ,
            '/Users/tiago/Documents/FCT/Tese/files_for_tiago/data/0002/kspaces/meas_MID00039_FID701346_T2_TSE_tra_obl-out_2.mrd'
            , 2, 12,20,0,0]]


#----------------- CODE -------------------

def processing_patient(file_paths, crop_protate_slices = False, crop_protate_in_xy_plane = False):

    o1, o2, patient, start, stop, offy, offx = file_paths[0], file_paths[1], file_paths[2], file_paths[3], file_paths[4], file_paths[5], file_paths[6]

    print("Patient", patient, "start")

    dset2 = ismrmrd.Dataset(o2, create_if_needed=False)
    dset = ismrmrd.Dataset(o1, create_if_needed=False)

    header2 = ismrmrd.xsd.CreateFromDocument(dset2.read_xml_header())
    header = ismrmrd.xsd.CreateFromDocument(dset.read_xml_header())

    enc = header.encoding[0]
    enc2 = header2.encoding[0]

    firstacq = 0

    for acqnum in range(dset.number_of_acquisitions()):
        acq = dset.read_acquisition(acqnum)

        if acq.isFlagSet(ismrmrd.ACQ_IS_NOISE_MEASUREMENT):
            continue
        else:
            firstacq = acqnum
            break

    ncoils = header.acquisitionSystemInformation.receiverChannels
    if enc.encodingLimits.slice != None:
        nslices = enc.encodingLimits.slice.maximum + 1
    else:
        nslices = 1

    if enc.encodingLimits.repetition != None:
        nreps = enc.encodingLimits.repetition.maximum + 1
    else:
        nreps = 1

    if enc.encodingLimits.contrast != None:
        ncontrasts = enc.encodingLimits.contrast.maximum + 1
    else:
        ncontrasts = 1

    eNx = enc.encodedSpace.matrixSize.x
    eNy = enc.encodedSpace.matrixSize.y
    eNz = enc.encodedSpace.matrixSize.z

    print("K-space Data for Patient ", patient, ' start')

    all_data = rf.sample_data_ET_mask(dset, nreps, ncontrasts, nslices, ncoils, eNz, eNy, eNx, firstacq = firstacq)

    all_z = [dset2.read_acquisition(n)._head.position[2] for n in range(dset2.number_of_acquisitions())]
    z = sorted(set(all_z))

    print("High Resolution Data for Patient ", patient, ' start')

    coil_aliase2, coil_aliase4, coil_aliase8, img = coil_aliased_aio(o2)

    print(np.shape(img))

    print("Low Resolution Data for Patient ", patient, ' start')

    if not crop_protate_slices:
        start = 0
        stop  = nslices

    for s in range(enc2.encodingLimits.slice.maximum + 1):
        print("Slice ", s, " of Patient ", patient, ' start')
        for n in range(dset.number_of_acquisitions()):
            if all_z[n] == z[s]:  # Find one acquisition of the wanted slice.
                img_pos = np.array(dset2.read_acquisition(n)._head.position)
                break

        offset_y = img_pos[1]
        offset_x = img_pos[0]

        im = np.zeros_like(all_data[0, 0, 0])

        for c in range(ncoils):
            im[c] = rf.transform_kspace_to_image(all_data[0, 0, 0, c, :, :, :])
            print("Coil ", c, " of Slice ", s, " of Patient ", patient, ' done')

        im = np.rollaxis(im, 3, 1)
        im = np.flip(im, 3)

        coil_images = rf.crop_array(im, 512, 512, offset_x, offset_y)

        find = np.argmin(abs(np.linspace(-500, 500, 128) - z[s]))

        coil_images_test = np.swapaxes(coil_images[:, find], 0, 2)
        print("Conv2D on going...")
        csm_est_test = rf.ismrm_estimate_csm_walsh(coil_images_test, 100)
        print("Conv2D Done")
        csm_est_test = np.swapaxes(csm_est_test, 0, 2)

        (unmix_sense2, gmap_sense) = rf.calculate_sense_unmixing(2,csm_est_test,regularization_factor = 1)
        (unmix_sense4, gmap_sense) = rf.calculate_sense_unmixing(4,csm_est_test,regularization_factor = 1)
        (unmix_sense8, gmap_sense) = rf.calculate_sense_unmixing(8,csm_est_test,regularization_factor = 1)

        recon_sense2 = rf.norm(np.squeeze(np.sum(coil_aliase2[s] * unmix_sense2,0)))
        recon_sense4 = rf.norm(np.squeeze(np.sum(coil_aliase4[s] * unmix_sense4,0)))
        recon_sense8 = rf.norm(np.squeeze(np.sum(coil_aliase8[s] * unmix_sense8,0)))



        ssim, rmse, vifp, ms_ssim, psnr = rf.iqm(rf.normalize8(np.abs(img[s])),
                                                 rf.normalize8(np.abs(recon_sense2)),
                                                 string=False)
        ssim4, rmse4, vifp4, ms_ssim4, psnr4 = rf.iqm(rf.normalize8(np.abs(img[s])),
                                                      rf.normalize8(np.abs(recon_sense4)),
                                                      string=False)
        ssim8, rmse8, vifp8, ms_ssim8, psnr8 = rf.iqm(rf.normalize8(np.abs(img[s])),
                                                      rf.normalize8(np.abs(recon_sense8)),
                                                      string=False)
        if crop_protate_in_xy_plane:
            ssim,rmse,vifp,ms_ssim,psnr = rf.iqm(rf.crop_array(rf.normalize8(np.abs(img[s])),70,70,offy,offx))
            ssim4,rmse4,vifp4,ms_ssim4, psnr4 = rf.iqm(rf.crop_array(rf.normalize8(np.abs(img[s])),70,70,offy,offx))
            ssim8,rmse8,vifp8,ms_ssim8, psnr8 = rf.iqm(rf.crop_array(rf.normalize8(np.abs(img[s])),70,70,offy,offx))



        name2 = 'SENSE_2' + '_' + str(patient) + '_' + str(s) + '_' + str(ssim) + '_' + str(vifp)+ '_' + str(ms_ssim)
        name4 = 'SENSE_4' + '_' + str(patient) + '_' + str(s) + '_' + str(ssim4) + '_' + str(vifp4)+ '_' + str(ms_ssim4)
        name8 = 'SENSE_8' + '_' + str(patient) + '_' + str(s) + '_' + str(ssim8) + '_' + str(vifp8)+ '_' + str(ms_ssim8)

        # Since image_name is a given argument, rf.imshow is not going to show the image but save it instead!

        rf.imshow(np.array([rf.normalize8(np.abs(img[s])), rf.normalize8(np.abs(recon_sense2))]), cmap="gray", tile_shape=(1, 2),
               colorbar=False,
               titles=['Ground Truth', 'SENSE2'], size=[100, 60], image_name = name2)

        rf.imshow(np.array([rf.normalize8(np.abs(img[s])), rf.normalize8(np.abs(recon_sense4))]), cmap="gray", tile_shape=(1, 2),
               colorbar=False,
               titles=['Ground Truth', 'SENSE4'], size=[100, 60], image_name= name4)

        rf.imshow(np.array([rf.normalize8(np.abs(img[s])), rf.normalize8(np.abs(recon_sense8))]), cmap="gray", tile_shape=(1, 2), colorbar=False,
               titles=['Ground Truth', 'SENSE8'], size=[100, 60], image_name = name8)


        data2 = ['SENSE', 2, patient, s, ssim, rmse, vifp, ms_ssim, psnr]
        data4 = ['SENSE', 4, patient, s, ssim4, rmse4, vifp4, ms_ssim4, psnr4]
        data8 = ['SENSE', 8, patient, s, ssim8, rmse8, vifp8, ms_ssim8, psnr8]

        '''
        with open('alldata_crop.csv', 'a', encoding='UTF8') as f:
            writer = csv.writer(f)
            writer.writerow(data2)
            writer.writerow(data4)
            writer.writerow(data8)
        f.close
        '''
        print("Slice ", s, " of Patient ", patient, ' Done!')
    print("Patient ", patient, " Done!")


if __name__ == "__main__":

    # Pool number is the number of parallel patients you wnat to work at the same time, one per each CPU core.
    # Remember that cores is not the only thing needed to increase this number, also memory is needed. It is recommended
    # to try 2 patients and monitorize the RAM usage. One patient already used 14gb in my case.
    with Pool(1) as pool:
        pool.map(processing_patient, args)
    print("Done")
