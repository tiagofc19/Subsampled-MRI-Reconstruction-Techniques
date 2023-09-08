# -*- coding: utf-8 -*-

#Basic setup
import ismrmrd.xsd
import csv
import numpy as np
import reconstruction_functions as rf
from multiprocessing import Pool

# args is a list of lists. Each list is a patient, with objects: Path for Dataset 1, Path for Dataset 2, Patient index,
# First slice, Last Slice + 1, Offset y, Offset x.
# Last 4 elements are not necessary if it is not planned to crop only the prostate area. Values won't be used and can be
# any value. Those values were chosen visually looking at slices plotted.
args = [[
            '/Users/tiago/Documents/FCT/Tese/files_for_tiago/data/0004/kspaces/meas_MID00460_FID707792_T2_TSE_tra_obl-out_1.mrd',
            '/Users/tiago/Documents/FCT/Tese/files_for_tiago/data/0004/kspaces/meas_MID00460_FID707792_T2_TSE_tra_obl-out_2.mrd',
            4,5,12,5,30], [
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


def processing_patient(file_paths, crop_protate_slices = False, crop_protate_in_xy_plane = False):

    o1, o2, patient, start, stop, offy, offx = file_paths[0], file_paths[1], file_paths[2], file_paths[3], file_paths[4], file_paths[5], file_paths[6]

    print("Patient", patient, "start")

    dset = ismrmrd.Dataset(o2, create_if_needed=False)

    header = ismrmrd.xsd.CreateFromDocument(dset.read_xml_header())
    enc = header.encoding[0]
    eNy = enc.encodedSpace.matrixSize.y
    eNz = enc.encodedSpace.matrixSize.z
    rNx = enc.reconSpace.matrixSize.x

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


    firstacq = 0
    for acqnum in range(dset.number_of_acquisitions()):
        acq = dset.read_acquisition(acqnum)

        if acq.isFlagSet(ismrmrd.ACQ_IS_NOISE_MEASUREMENT):
            continue
        else:
            firstacq = acqnum
            break


    all_data = np.zeros((nslices, ncoils, eNz, eNy + 1, rNx), dtype=np.complex64)
    r2 = np.zeros((nslices, ncoils, eNz, eNy + 1, rNx), dtype=np.complex64)
    r4 = np.zeros((nslices, ncoils, eNz, eNy + 1, rNx), dtype=np.complex64)
    r8 = np.zeros((nslices, ncoils, eNz, eNy + 1, rNx), dtype=np.complex64)

    av1 = np.zeros((nslices, ncoils, eNz, eNy + 1, rNx), dtype=np.complex64)
    av2 = np.zeros((nslices, ncoils, eNz, eNy + 1, rNx), dtype=np.complex64)
    av3 = np.zeros((nslices, ncoils, eNz, eNy + 1, rNx), dtype=np.complex64)

    print("K-space Data for Patient ", patient, ' start')


    for acqnum in range(firstacq, dset.number_of_acquisitions()):

        acq = dset.read_acquisition(acqnum)

        # Stuff into the buffer

        slice = acq.idx.slice
        y = acq.idx.kspace_encode_step_1
        z = acq.idx.kspace_encode_step_2

        av = acq._head.idx.average + 1
        if av == 1 or av == 2:  # only getting the first two averages
            all_data[slice, :, z, y, :] = acq.data
            r2[slice, :, z, y, :] = acq.data
            r4[slice, :, z, y, :] = acq.data
            r8[slice, :, z, y, :] = acq.data


        #if av == 1:
        #    av1[rep, contrast, slice, :, z, y, :] = acq.data

    all_data = rf.image_order(all_data)
    all_data = rf.crop_array(all_data, 512, 476)
    calib = all_data[:, :, 0, 335:355, :]
    r2 = rf.image_order(r2)
    r4 = rf.image_order(r4)
    r8 = rf.image_order(r8)


    r2 = rf.homogeneous_mask(r2, 2)
    r2 = rf.crop_array(r2, 512, 476)
    r4 = rf.homogeneous_mask(r4, 4)
    r4 = rf.crop_array(r4, 512, 476)
    r8 = rf.homogeneous_mask(r8, 8)
    r8 = rf.crop_array(r8, 512, 476)


    print("K-space Data for Patient ", patient, ' done')


    nslices = enc.encodingLimits.slice.maximum + 1

    if not crop_protate_slices:
        start = 0
        stop = nslices

    for slice in range(start,stop):  # 31
        print("Slice ", slice, " of Patient ", patient, ' Start!')

        grappa2 = rf.grappa(r2[slice, :, 0], calib[slice], kernel_size=(3, 3), coil_axis=0)
        print('R2 DONE')
        grappa4 = rf.grappa(r4[slice, :, 0], calib[slice], kernel_size=(5, 5), coil_axis=0)
        print('R4 DONE')
        grappa8 = rf.grappa(r8[slice, :, 0], calib[slice], kernel_size=(9, 9), coil_axis=0)

        im = rf.transform_kspace_to_image(all_data[slice, :, 0, :, :], [1, 2])
        im2 = rf.transform_kspace_to_image(grappa2, [1, 2])
        im4 = rf.transform_kspace_to_image(grappa4, [1, 2])
        im8 = rf.transform_kspace_to_image(grappa8, [1, 2])

        coil_images_1 = rf.pad_image_stack(im)
        coil_images_2 = rf.pad_image_stack(im2)
        coil_images_4 = rf.pad_image_stack(im4)
        coil_images_8 = rf.pad_image_stack(im8)

        grappa_img2 = np.sqrt(np.sum(np.abs(coil_images_2) ** 2, 0))
        grappa_img4 = np.sqrt(np.sum(np.abs(coil_images_4) ** 2, 0))
        grappa_img8 = np.sqrt(np.sum(np.abs(coil_images_8) ** 2, 0))
        im = np.sqrt(np.sum(np.abs(coil_images_1) ** 2, 0))

        if crop_protate_in_xy_plane:
            im = rf.crop_array(rf.normalize8(np.abs(im)), 70, 70, offy, offx)
            grappa_img2 = rf.crop_array(rf.normalize8(np.abs(grappa_img2)), 70, 70, offy, offx)
            grappa_img4 = rf.crop_array(rf.normalize8(np.abs(grappa_img4)), 70, 70, offy, offx)
            grappa_img8 = rf.crop_array(rf.normalize8(np.abs(grappa_img8)), 70, 70, offy, offx)

        ssim, rmse, vifp, ms_ssim, psnr = rf.iqm(im, grappa_img2,
                                              string=False)
        ssim4, rmse4, vifp4, ms_ssim4, psnr4 = rf.iqm(im, grappa_img4,
                                                   string=False)
        ssim8, rmse8, vifp8, ms_ssim8, psnr8 = rf.iqm(im, grappa_img8,
                                                   string=False)

        data2 = ['GRAPPA', 2, patient, slice, ssim, rmse, vifp, ms_ssim, psnr]
        data4 = ['GRAPPA', 4, patient, slice, ssim4, rmse4, vifp4, ms_ssim4, psnr4]
        data8 = ['GRAPPA', 8, patient, slice, ssim8, rmse8, vifp8, ms_ssim8, psnr8]

        name2 = 'GRAPPA_2' + '_' + str(patient) + '_' + str(slice) + '_' + str(ssim) + '_' + str(vifp)+ '_' + str(ms_ssim)
        name4 = 'GRAPPA_4' + '_' + str(patient) + '_' + str(slice) + '_' + str(ssim4) + '_' + str(vifp4)+ '_' + str(ms_ssim4)
        name8 = 'GRAPPA_8' + '_' + str(patient) + '_' + str(slice) + '_' + str(ssim8) + '_' + str(vifp8)+ '_' + str(ms_ssim8)


        rf.imshow(np.array([rf.normalize8(np.abs(im)), rf.normalize8(np.abs(grappa_img2))]), cmap="gray", tile_shape=(1, 2),
               colorbar=False,
               titles=['Ground Truth', 'GRAPPA2'], size=[100, 60], image_name = name2)
        
        rf.imshow(np.array([rf.normalize8(np.abs(im)), rf.normalize8(np.abs(grappa_img4))]), cmap="gray", tile_shape=(1, 2),
               colorbar=False,
               titles=['Ground Truth', 'GRAPPA2'], size=[100, 60], image_name = name4)
        
        rf.imshow(np.array([rf.normalize8(np.abs(im)), rf.normalize8(np.abs(grappa_img8))]), cmap="gray", tile_shape=(1, 2), colorbar=False,
               titles=['Ground Truth', 'GRAPPA2'], size=[100, 60], image_name = name8)


        '''
        with open('alldata_grappa.csv', 'a', encoding='UTF8') as f:
            writer = csv.writer(f)
            writer.writerow(data2)
            writer.writerow(data4)
            writer.writerow(data8)
        f.close
        '''

        print("Slice ", slice, " of Patient ", patient, ' Done!')
    print("Patient ", patient, " Done!")

if __name__ == "__main__":

    # Pool number is the number of parallel patients you wnat to work at the same time, one per each CPU core.
    # Remember that cores is not the only thing needed to increase this number, also memory is needed. It is recommended
    # to try 2 patients and monitorize the RAM usage. One patient already used 14gb in my case.
    with Pool(1) as pool:
      pool.map(processing_patient, args)
    print("Done")