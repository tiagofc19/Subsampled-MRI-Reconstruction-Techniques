### IMPORTS
from sewar.full_ref import *
import numpy as np
import matplotlib.image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import RectangleSelector
from time import time
from tempfile import NamedTemporaryFile as NTF
from skimage.util import view_as_windows
from scipy.signal import convolve2d
from scipy.ndimage import gaussian_filter
import ismrmrd
from scipy import ndimage
from scipy.fft import fftn, ifftn, fftshift, ifftshift


### SECTION 1

def echo_train_length(dset) -> int:
    """
    Calculate the echo train length (ETL) from an ISMRMRD dataset.

    Args:
    - dset: ISMRMRD dataset object containing acquisitions.

    Returns:
    - ETL: Echo train length if found, or None if not found.
    """
    # This assumes:
    # The noise acquisitions are made in the beggining.
    # The noise acquisition is made in the same slice as the first ET.
    # There are multiple slices.
    # After the first ET, it moves to the first ET of a different slice.
    # The first ET has the same leght as the other ones.

    # Loop through acquisitions to find the starting point of the first echo train
    for n in range(dset.number_of_acquisitions()):
        if dset.read_acquisition(n).isFlagSet(ismrmrd.ACQ_IS_NOISE_MEASUREMENT):
            continue
        else:
            first = n
            break

    # Loop through acquisitions again to find the change in slice index
    for n in range(dset.number_of_acquisitions()):
        if dset.read_acquisition(n)._head.idx.slice != dset.read_acquisition(0)._head.idx.slice:
            return n - first

    # If different slices are not found, print a message and return None
    print("Couldn't find different slices in the dataset")
    return None


def echo_train_count(dset):
    """
    Calculate the echo train count (ETC) from an ISMRMRD dataset.

    Args:
    - dset: ISMRMRD dataset object containing acquisitions.

    Returns:
    - ETC: Echo train count if found, or None if not found.
    """
    #This assumes:
    # All the assumptions of ech_train_length
    # There are at least 2 averages (idx 0 and 1)
    # Higher index averages are acquired later

    # Get the encoding limits from the XML header
    enc = ismrmrd.xsd.CreateFromDocument(dset.read_xml_header()).encoding[0]

    if enc.encodingLimits.slice is not None:
        nslices = enc.encodingLimits.slice.maximum + 1
    else:
        print("Couldn't find different slices in the dataset")
        return None

    count = 0

    # Loop through acquisitions to count the number of averages
    for n in range(dset.number_of_acquisitions()):
        if dset.read_acquisition(n)._head.idx.average == 2:
            break
        if dset.read_acquisition(n)._head.idx.average == 1:
            count += 1

    # Calculate and return the echo train count
    etc = int(count / (nslices * echo_train_length(dset)))
    return etc

def normalize8(image):
    """
    Normalize an input image array to the range [0, 255] and convert to 8-bit unsigned integers.

    Parameters:
    image (numpy.ndarray): Input image array containing numeric pixel values to be normalized.

    Returns:
    numpy.ndarray: Normalized image array with pixel values in the range [0, 255] as 8-bit unsigned integers.
    """

    # Find the minimum pixel value in the input image array
    min_pixel = image.min()

    # Find the maximum pixel value in the input image array
    max_pixel = image.max()

    # Calculate the range of pixel values
    pixel_range = max_pixel - min_pixel

    # Ensure we don't divide by zero
    if pixel_range == 0:
        raise ValueError("Input image has constant pixel values, cannot normalize.")

    # Normalize the image's pixel values to the range [0, 255]
    # by subtracting the minimum value and then scaling to 8-bit range
    normalized_image = ((image - min_pixel) / pixel_range) * 255

    # Convert the normalized image to 8-bit unsigned integers
    normalized_image_uint8 = normalized_image.astype(np.uint8)

    return normalized_image_uint8


def iqm(im1, im2, string=True):
    """
    Calculate and return various image quality metrics between two input images.

    Parameters:
    im1 (numpy.ndarray): The first input image for comparison.
    im2 (numpy.ndarray): The second input image for comparison.
    string (bool): If True, return the metrics as a formatted string.
                   If False, return the metrics as a tuple of numeric values.

    Returns:
    str or tuple: If string=True, a formatted string containing SSIM, RMSE, VIFP, MS-SSIM, and PSNR metrics.
                  If string=False, a tuple containing SSIM, RMSE, VIFP, MS-SSIM, and PSNR values.
    """
    # Calculate SSIM, RMSE, VIFP, MS-SSIM, and PSNR metrics
    ssim_score = ssim(im1, im2)[0]
    rmse_score = np.sqrt(mse(im1, im2))
    vifp_score = vifp(im1, im2)
    msssim_score = np.real(msssim(im1, im2))
    psnr_score = psnr(im1, im2)

    if string:
        # Return metrics as a formatted string
        metrics_string = f'SSIM: {ssim_score}\nRMSE: {rmse_score}\nVIFP: {vifp_score}\nMS-SSIM: {msssim_score}\nPSNR: {psnr_score}'
        return metrics_string
    else:
        # Return metrics as a tuple of numeric values
        return ssim_score, rmse_score, vifp_score, msssim_score, psnr_score


def homogeneous_mask(data, R):
    """
    Create a homogeneous mask for a given 4D data array by setting values at regular intervals to zero.

    Parameters:
    data (numpy.ndarray): Input 4D data array.
    R (int): Interval for setting values to zero. For example, if R=2, every second value will be set to zero.

    Returns:
    numpy.ndarray: 4D data array with values set to zero at regular intervals defined by R.
    """
    masked_data = data.copy()
    # Create a mask with the same shape as the input data, initialized with zeros
    mask = np.zeros_like(data)

    # Set values to 1 in the mask at regular intervals along the specified dimension
    mask[:, :, :, ::R, :] = 1

    # Set values in the input data to zero where the mask is zero
    masked_data[mask == 0] = 0

    return masked_data


def zero_padding_zy(data,a=2.0):
    sh_z, sh_y = [np.shape(data)[2],np.shape(data)[3]]
    im_z, im_y = int(a * sh_z), int(a * sh_y)
    z_0 = im_z // 2 - (sh_z // 2)
    y_0 = im_y // 2 - (sh_y // 2)

    y3_shape = list(data.shape)
    y3_shape[2] = int(a * y3_shape[2])
    y3_shape[3] = int(a * y3_shape[3])
    y3 = np.zeros(y3_shape, dtype=np.complex64)

    y3[:, :, z_0:z_0 + sh_z, y_0:y_0 + sh_y, :] = data

    return y3


def image_order(data):
    """
    Rearrange the data array to a specific order by interleaving two halves of the data along a dimension.

    Parameters:
    data (numpy.ndarray): Input data array to be rearranged.

    Returns:
    numpy.ndarray: Rearranged data array with two halves interleaved along a dimension.
    """
    # Create a new complex array with the same shape as the input data, initialized with zeros
    ordered_data = np.zeros_like(data, dtype=np.complex64)

    # Calculate the number of rows for the first half of the data
    half = ordered_data[::2, :, :].shape[0]

    # Interleave the two halves of the data along the specified dimension
    ordered_data[::2] = data[:half]
    ordered_data[1::2] = data[half:]

    return ordered_data


def set_size(w, h, ax=None):
    """
    Set the size of a matplotlib figure.

    Args:
    - w: Width of the figure in inches.
    - h: Height of the figure in inches.
    - ax: Axes object for which the figure size needs to be set (optional).

    Notes:
    - If `ax` is not provided, the current Axes object is obtained using `plt.gca()`.
    - The function calculates the figure size based on the provided width and height,
      as well as the subplot parameters (left, right, top, bottom) of the Axes object.
      It then sets the figure size using the calculated values.
    """
    if not ax:
        ax = plt.gca()

    # Get the subplot parameters of the Axes object
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom

    # Calculate the figure width and height based on desired values and subplot parameters
    figw = float(w) / (r - l)
    figh = float(h) / (t - b)

    # Set the figure size
    ax.figure.set_size_inches(figw, figh)


def imshow(image_matrix, tile_shape=None, scale=None, titles=[], fontsize = 100, colorbar=False, cmap='jet', size = [10,10], text = '', image_name = ''):

    #Added some changes on matplotlib.pyplot.imshow, ref: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imshow.html

    """ Tiles images and displays them in a window.

    :param image_matrix: a 2D or 3D set of image data
    :param tile_shape: optional shape ``(rows, cols)`` for tiling images
    :param scale: optional ``(min,max)`` values for scaling all images
    :param titles: optional list of titles for each subplot
    :param cmap: optional colormap for all images
    """

    assert image_matrix.ndim in [2, 3], "image_matrix must have 2 or 3 dimensions"

    if image_matrix.ndim == 2:
        image_matrix = image_matrix.reshape((1, image_matrix.shape[0], image_matrix.shape[1]))

    if not scale:
        scale = (np.min(image_matrix), np.max(image_matrix))
    vmin, vmax = scale

    if not tile_shape:
        tile_shape = (1, image_matrix.shape[0])
    assert np.prod(tile_shape) >= image_matrix.shape[0],\
        "image tile rows x columns must equal the 3rd dim extent of image_matrix"

    # add empty titles as necessary
    if len(titles) < image_matrix.shape[0]:
        titles.extend(['' for x in range(image_matrix.shape[0] - len(titles))])

    if len(titles) > 0:
        assert len(titles) >= image_matrix.shape[0],\
                "number of titles must equal 3rd dim extent of image_matrix"

    def onselect(eclick, erelease):
        print((eclick.xdata, eclick.ydata), (erelease.xdata, erelease.ydata))

    def on_pick(event):
        if isinstance(event.artist, matplotlib.image.AxesImage):
            x, y = event.mouseevent.xdata, event.mouseevent.ydata
            im = event.artist
            A = im.get_array()
            print(A[y, x])

    selectors = [] # need to keep a reference to each selector
    rectprops = dict(facecolor='red', edgecolor='black', alpha=0.5, fill=True)
    cols, rows = tile_shape
    fig = plt.figure()
    gs1 = gridspec.GridSpec(cols, rows)
    gs1.update(wspace=0.025, hspace=0.025)
    plt.text(0,0, text, fontsize = fontsize)
    plt.set_cmap(cmap)
    plt.axis('off')
    for z in range(image_matrix.shape[0]):
        ax = fig.add_subplot(gs1[z])
        ax.set_title(titles[z], size= fontsize)
        ax.set_axis_off()
        ax.set_aspect('auto')
        imgplot = ax.imshow(image_matrix[z,:,:], vmin=vmin, vmax=vmax, picker=True)
        selectors.append(RectangleSelector(ax, onselect))

        if colorbar is True:
            plt.colorbar(imgplot, shrink= 0.8)

    fig.canvas.callbacks.connect('pick_event', on_pick)
    x1, y1 = size
    set_size(x1,y1)
    if image_name == '':
        plt.show()
    else:
        plt.savefig(image_name + '.png')


def crop_array(arr, x, y, offx=0, offy=0):
    """
    Crop the last two dimensions of a multidimensional array to the size of x,y.

    Args:
    - arr: numpy array of shape (..., H, W), where H and W are the height and width of the last two dimensions.
    - x: integer representing the desired height of the cropped array.
    - y: integer representing the desired width of the cropped array.
    - offx: optional integer representing the vertical offset (default is 0).
    - offy: optional integer representing the horizontal offset (default is 0).

    Returns:
    - cropped_arr: numpy array of shape (..., x, y), representing the cropped version of the input array.
    """
    # Calculate pixel offsets based on the provided offsets and scaling factor
    # This is used to crop dataset 1, offset information is in each slice of dataset 2 in millimeters, so we need to
    # convert to pixels by dividing by the FOV (mm) and multiplying by the FOV (px).
    offx_px = int(512 * offx / 386.64)
    offy_px = int(512 * offy / 386.64)

    # Get the current size of the last two dimensions
    curr_h, curr_w = arr.shape[-2:]

    # Compute the indices of the center crop
    start_h = ((curr_h - x) // 2)
    end_h = start_h + x
    start_w = ((curr_w - y) // 2)
    end_w = start_w + y

    # Crop the array using the calculated indices and offsets
    cropped_arr = arr[..., start_h - offx_px:end_h - offx_px, start_w - offy_px:end_w - offy_px]

    return cropped_arr


def pad_image_stack(images):
    """
    Pad a stack of 2D images to a common size of 512x512.

    Args:
    - images: List or array of 2D image arrays.

    Returns:
    - padded_images: Array of padded 2D image arrays with a common size of 512x512.
    """
    # Initialize an array to store padded images
    padded_images = np.zeros((len(images), 512, 512), dtype=complex)

    # Iterate over each image and pad it to 512x512
    for i, image in enumerate(images):
        rows, cols = image.shape

        # Calculate padding values for each side
        left_pad = (512 - cols) // 2
        right_pad = 512 - cols - left_pad
        top_pad = (512 - rows) // 2
        bottom_pad = 512 - rows - top_pad

        # Pad the image using calculated padding values
        padded_image = np.pad(image, ((top_pad, bottom_pad), (left_pad, right_pad)), mode='constant')

        # Store the padded image in the array
        padded_images[i] = padded_image

    return padded_images


def grappa(kspace, calib, kernel_size=(5, 5), coil_axis=-1, lamda=0.01, memmap=False, memmap_filename='out.memmap', silent=True):

    # SOURCE: https://github.com/mckib2/pygrappa/blob/688e845e81fd37dd0632e135c7ab66caa6b2b7d6/pygrappa/grappa.py#L32

    '''GeneRalized Autocalibrating Partially Parallel Acquisitions.

    Parameters
    ----------
    kspace : array_like
        2D multi-coil k-space data to reconstruct from.  Make sure
        that the missing entries have exact zeros in them.
    calib : array_like
        Calibration data (fully sampled k-space).
    kernel_size : tuple, optional
        Size of the 2D GRAPPA kernel (kx, ky).
    coil_axis : int, optional
        Dimension holding coil data.  The other two dimensions should
        be image size: (sx, sy).
    lamda : float, optional
        Tikhonov regularization for the kernel calibration.
    memmap : bool, optional
        Store data in Numpy memmaps.  Use when datasets are too large
        to store in memory.
    memmap_filename : str, optional
        Name of memmap to store results in.  File is only saved if
        memmap=True.
    silent : bool, optional
        Suppress messages to user.

    Returns
    -------
    res : array_like
        k-space data where missing entries have been filled in.

    Notes
    -----
    Based on implementation of the GRAPPA algorithm [1]_ for 2D
    images.

    If memmap=True, the results will be written to memmap_filename
    and nothing is returned from the function.

    References
    ----------
    .. [1] Griswold, Mark A., et al. "Generalized autocalibrating
           partially parallel acquisitions (GRAPPA)." Magnetic
           Resonance in Medicine: An Official Journal of the
           International Society for Magnetic Resonance in Medicine
           47.6 (2002): 1202-1210.
    '''

    # Remember what shape the final reconstruction should be
    fin_shape = kspace.shape[:]

    # Put the coil dimension at the end
    kspace = np.moveaxis(kspace, coil_axis, -1)
    calib = np.moveaxis(calib, coil_axis, -1)

    # Quit early if there are no holes
    if np.sum((np.abs(kspace[..., 0]) == 0).flatten()) == 0:
        return np.moveaxis(kspace, -1, coil_axis)

    # Get shape of kernel
    kx, ky = kernel_size[:]
    kx2, ky2 = int(kx / 2), int(ky / 2)
    nc = calib.shape[-1]

    # When we apply weights, we need to select a window of data the
    # size of the kernel.  If the kernel size is odd, the window will
    # be symmetric about the target.  If it's even, then we have to
    # decide where the window lies in relation to the target.  Let's
    # arbitrarily decide that it will be right-sided, so we'll need
    # adjustment factors used as follows:
    #     S = kspace[xx-kx2:xx+kx2+adjx, yy-ky2:yy+ky2+adjy, :]
    # Where:
    #     xx, yy : location of target
    adjx = np.mod(kx, 2)
    adjy = np.mod(ky, 2)

    # Pad kspace data
    kspace = np.pad(  # pylint: disable=E1102
        kspace, ((kx2, kx2), (ky2, ky2), (0, 0)), mode='constant')
    calib = np.pad(  # pylint: disable=E1102
        calib, ((kx2, kx2), (ky2, ky2), (0, 0)), mode='constant')

    # Notice that all coils have same sampling pattern, so choose
    # the 0th one arbitrarily for the mask
    mask = np.ascontiguousarray(np.abs(kspace[..., 0]) > 0)
    # Store windows in temporary files so we don't overwhelm memory
    with NTF() as fP, NTF() as fA, NTF() as frecon:

        # Start the clock...
        t0 = time()

        # Get all overlapping patches from the mask
        P = np.memmap(fP, dtype=mask.dtype, mode='w+', shape=(
            mask.shape[0] - 2 * kx2, mask.shape[1] - 2 * ky2, 1, kx, ky))
        P = view_as_windows(mask, (kx, ky))
        Psh = P.shape[:]  # save shape for unflattening indices later
        P = P.reshape((-1, kx, ky))

        # Find the unique patches and associate them with indices
        P, iidx = np.unique(P, return_inverse=True, axis=0)

        # Filter out geometries that don't have a hole at the center.
        # These are all the kernel geometries we actually need to
        # compute weights for.
        validP = np.argwhere(~P[:, kx2, ky2]).squeeze()

        # We also want to ignore empty patches
        invalidP = np.argwhere(np.all(P == 0, axis=(1, 2)))
        validP = np.setdiff1d(validP, invalidP, assume_unique=True)

        # Make sure validP is iterable
        validP = np.atleast_1d(validP)

        # Give P back its coil dimension
        P = np.tile(P[..., None], (1, 1, 1, nc))

        if not silent:
            print('P took %g seconds!' % (time() - t0))
        t0 = time()

        # Get all overlapping patches of ACS
        try:
            A = np.memmap(fA, dtype=calib.dtype, mode='w+', shape=(
                calib.shape[0] - 2 * kx, calib.shape[1] - 2 * ky, 1, kx, ky, nc))
            A[:] = view_as_windows(
                calib, (kx, ky, nc)).reshape((-1, kx, ky, nc))
        except ValueError:
            A = view_as_windows(
                calib, (kx, ky, nc)).reshape((-1, kx, ky, nc))

        # Report on how long it took to construct windows
        if not silent:
            print('A took %g seconds' % (time() - t0))

        # Initialize recon array
        recon = np.memmap(
            frecon, dtype=kspace.dtype, mode='w+',
            shape=kspace.shape)

        # Train weights and apply them for each valid hole we have in
        # kspace data:
        t0 = time()
        for ii in validP:
            # Get the sources by masking all patches of the ACS and
            # get targets by taking the center of each patch. Source
            # and targets will have the following sizes:
            #     S : (# samples, N possible patches in ACS)
            #     T : (# coils, N possible patches in ACS)
            # Solve the equation for the weights:
            #     WS = T
            #     WSS^H = TS^H
            #  -> W = TS^H (SS^H)^-1
            # S = A[:, P[ii, ...]].T # transpose to get correct shape
            # T = A[:, kx2, ky2, :].T
            # TSh = T @ S.conj().T
            # SSh = S @ S.conj().T
            # W = TSh @ np.linalg.pinv(SSh) # inv won't work here

            # Equivalenty, we can formulate the problem so we avoid
            # computing the inverse, use numpy.linalg.solve, and
            # Tikhonov regularization for better conditioning:
            #     SW = T
            #     S^HSW = S^HT
            #     W = (S^HS)^-1 S^HT
            #  -> W = (S^HS + lamda I)^-1 S^HT
            # Notice that this W is a transposed version of the
            # above formulation.  Need to figure out if W @ S or
            # S @ W is more efficient matrix multiplication.
            # Currently computing W @ S when applying weights.
            S = A[:, P[ii, ...]]
            T = A[:, kx2, ky2, :]
            ShS = S.conj().T @ S
            ShT = S.conj().T @ T
            lamda0 = lamda * np.linalg.norm(ShS) / ShS.shape[0]
            W = np.linalg.solve(
                ShS + lamda0 * np.eye(ShS.shape[0]), ShT).T

            # Now that we know the weights, let's apply them!  Find
            # all holes corresponding to current geometry.
            # Currently we're looping through all the points
            # associated with the current geometry.  It would be nice
            # to find a way to apply the weights to everything at
            # once.  Right now I don't know how to simultaneously
            # pull all source patches from kspace faster than a
            # for loop...

            # x, y define where top left corner is, so move to ctr,
            # also make sure they are iterable by enforcing atleast_1d
            idx = np.unravel_index(
                np.argwhere(iidx == ii), Psh[:2])
            x, y = idx[0] + kx2, idx[1] + ky2
            x = np.atleast_1d(x.squeeze())
            y = np.atleast_1d(y.squeeze())
            for xx, yy in zip(x, y):
                # Collect sources for this hole and apply weights
                S = kspace[xx - kx2:xx + kx2 + adjx, yy - ky2:yy + ky2 + adjy, :]
                S = S[P[ii, ...]]
                recon[xx, yy, :] = (W @ S[:, None]).squeeze()

        # Report on how long it took to train and apply weights
        if not silent:
            print(('Training and application of weights took %g'
                   'seconds' % (time() - t0)))

        # The recon array has been zero padded, so let's crop it down
        # to size and return it either as a memmap to the correct
        # file or in memory.
        # Also fill in known data, crop, move coil axis back.
        if memmap:
            fin = np.memmap(
                memmap_filename, dtype=recon.dtype, mode='w+',
                shape=fin_shape)
            fin[:] = np.moveaxis(
                (recon + kspace)[kx2:-kx2, ky2:-ky2, :],
                -1, coil_axis)
            del fin
            return None

        return np.moveaxis((recon[:] + kspace)[kx2:-kx2, ky2:-ky2, :], -1, coil_axis)


def transform_kspace_to_image(k, dim=None, img_shape=None):
    """
    Computes the Fourier transform from k-space to image space along given or all dimensions.

    Args:
    - k: k-space data (complex-valued)
    - dim: Vector of dimensions to transform (optional, default is all dimensions).
    - img_shape: Desired shape of the output image (optional).

    Returns:
    - img: Data in image space along transformed dimensions.
    """
    if not dim:
        dim = range(k.ndim)

    # Apply inverse Fourier transform along specified dimensions
    k_shifted = ifftshift(k, axes=dim)
    img = ifftn(k_shifted, s=img_shape, axes=dim)
    img = fftshift(img, axes=dim)

    # Adjust the amplitude by the square root of the product of dimensions
    img *= np.sqrt(np.prod(np.take(img.shape, dim)))

    return img


def ismrm_eig_power(R):
    """
    Compute dominant eigenvectors using the power method.

    Args:
    - R: 4D array representing sample correlation matrices (rows x cols x ncoils x ncoils).

    Returns:
    - v: Dominant eigenvectors corresponding to the sample correlation matrices.
    - d: Dominant eigenvalues corresponding to the sample correlation matrices.
    """
    rows, cols, ncoils, _ = R.shape
    N_iterations = 2
    v = np.ones((rows, cols, ncoils))  # Initialize eigenvectors

    d = np.zeros((rows, cols))
    for i in range(N_iterations):
        # Calculate the matrix-vector product R*v along the last dimension
        v = np.sum(R * np.tile(v[:, :, :, np.newaxis], (1, 1, 1, ncoils)), axis=2)

        # Calculate the magnitude of the vector
        d = np.sqrt(np.sum(v * np.conj(v), axis=2))
        d[d <= np.finfo(float).eps] = np.finfo(float).eps

        # Normalize the vector by dividing it by its magnitude
        v = v / np.tile(d[:, :, np.newaxis], (1, 1, ncoils))

    # Calculate the phase of the first coil's conjugate
    p1 = np.angle(np.conj(v[:, :, 0]))

    # Optionally, normalize output to coil 1 phase
    v = v * np.tile(np.exp(1j * p1)[:, :, np.newaxis], (1, 1, ncoils))
    v = np.conj(v)  # Conjugate the eigenvectors
    return v, d


def ismrm_correlation_matrix(s):
    """
    Compute the sample correlation matrix for coil sensitivity maps.

    Args:
    - s: 3D array representing coil sensitivity maps (rows x cols x ncoils).

    Returns:
    - Rs: 4D array representing the sample correlation matrix (rows x cols x ncoils x ncoils).
    """
    rows, cols, ncoils = s.shape
    Rs = np.zeros((rows, cols, ncoils, ncoils), dtype=np.complex64)  # Initialize sample correlation matrix

    # Iterate over coil pairs and compute correlations
    for i in range(ncoils):
        for j in range(i):
            Rs[:, :, i, j] = s[:, :, i] * np.conj(s[:, :, j])
            Rs[:, :, j, i] = np.conj(Rs[:, :, i, j])  # Using conjugate symmetry of Rs
        Rs[:, :, i, i] = s[:, :, i] * np.conj(s[:, :, i])

    return Rs

def smooth(img, box=5):
    '''Smooths coil images

    :param img: Input complex images, ``[y, x] or [z, y, x]``
    :param box: Smoothing block size (default ``5``)

    :returns simg: Smoothed complex image ``[y,x] or [z,y,x]``
    '''

    t_real = np.zeros(img.shape)
    t_imag = np.zeros(img.shape)

    ndimage.filters.uniform_filter(img.real,size=box,output=t_real)
    ndimage.filters.uniform_filter(img.imag,size=box,output=t_imag)

    simg = t_real + 1j*t_imag

    return simg


def ismrm_estimate_csm_walsh(img, smoothing=5):

    # Python conversion of the funtions from https://github.com/hansenms/ismrm_sunrise_matlab/blob/master/ismrm_estimate_csm_walsh.m
    #Estimates relative coil sensitivity maps from a set of coil images using the eigenvector method described
    # by Walsh et al. (Magn Reson Med2000;43:682-90.)
    """
    Estimate coil sensitivity maps using the Walsh method.

    Args:
    - img: 3D array representing complex image data (height x width x ncoils).
    - smoothing: Integer representing the size of spatial smoothing window (optional, default is 5).

    Returns:
    - csm: Coil sensitivity maps estimated using the Walsh method.
    """

    ncoils = img.shape[2]

    # normalize by root sum of squares magnitude
    mag = np.sqrt(np.sum(img * np.conj(img), axis=2))
    s_raw = img / np.tile((mag + np.finfo(float).eps)[:, :, np.newaxis], (1, 1, ncoils))
    del mag

    # compute sample correlation estimates at each pixel location
    Rs = ismrm_correlation_matrix(s_raw)

    # apply spatial smoothing to sample correlation estimates (NxN convolution)
    if smoothing > 1:
        h_smooth = np.ones((smoothing, smoothing)) / (smoothing ** 2)  # uniform smoothing kernel
        for m in range(ncoils):
            for n in range(ncoils):
                # Rs[:, :, m, n] = convolve2d(Rs[:, :, m, n], h_smooth, mode='same')
                Rs[:, :, m, n] = smooth(Rs[:, :, m, n], smoothing)

    # compute dominant eigenvectors of sample correlation matrices
    csm, rho = ismrm_eig_power(Rs)  # using power method
    return csm


def _calculate_sense_unmixing_1d(acc_factor, csm1d, regularization_factor):
    """
    Calculate SENSE unmixing coefficients for 1D data.

    Args:
    - acc_factor: Acceleration factor.
    - csm1d: 2D array representing coil sensitivity maps (coils x data points).
    - regularization_factor: Regularization factor for stabilizing the inversion.

    Returns:
    - unmix1d: 2D array of SENSE unmixing coefficients.
    """
    nc = csm1d.shape[0]
    ny = csm1d.shape[1]

    assert (ny % acc_factor) == 0, "ny must be a multiple of the acceleration factor"

    unmix1d = np.zeros((nc, ny), dtype=np.complex64)

    nblocks = int(ny / acc_factor)
    for b in range(0, nblocks):
        A = np.matrix(csm1d[:, b:ny:nblocks]).T

        if np.max(np.abs(A)) > 0:
            # Calculate A^H * A
            AHA = A.H * A
            reduced_eye = np.diag(np.abs(np.diag(AHA)) > 0)
            n_alias = np.sum(reduced_eye)

            # Scale regularization factor based on trace of AHA and number of non-aliased points
            scaled_reg_factor = regularization_factor * np.trace(AHA) / n_alias

            # Calculate pseudoinverse using regularized inversion
            unmix1d[:, b:ny:nblocks] = np.linalg.pinv(AHA + (reduced_eye * scaled_reg_factor)) * A.H

    return unmix1d


def calculate_sense_unmixing(acc_factor, csm, regularization_factor = 0.001):

    # SOURCE: https://github.com/ismrmrd/ismrmrd-python-tools/blob/master/ismrmrdtools/coils.py
    '''Calculates the unmixing coefficients for a 2D image using a SENSE algorithm

    :param acc_factor: Acceleration factor, e.g. 2
    :param csm: Coil sensitivity map, ``[coil, y, x]``
    :param regularization_factor: adds tychonov regularization (default ``0.001``)

        - 0 = no regularization
        - set higher for more aggressive regularization.

    :returns unmix: Image unmixing coefficients for a single ``x`` location, ``[coil, y, x]``
    :returns gmap: Noise enhancement map, ``[y, x]``
    '''

    assert csm.ndim == 3, "Coil sensitivity map must have exactly 3 dimensions"

    unmix = np.zeros(csm.shape,np.complex64)

    for x in range(0,csm.shape[2]):
        unmix[:,:,x] = _calculate_sense_unmixing_1d(acc_factor, np.squeeze(csm[:,:,x]), regularization_factor)

    gmap = np.squeeze(np.sqrt(np.sum(abs(unmix) ** 2, 0))) * np.squeeze(np.sqrt(np.sum(abs(csm) ** 2, 0)))

    return (unmix,gmap)


def sample_data_ET_mask(dset, etl, etc, leave_ET=0):

    #This functions samples the data into a numpy array with the option of subsampling it removing echo trains.
    
    '''
    Arguments:
        - dset: data set in ismrmrd format
        - leave_ET: number of echo trains to remove from the end of the sampled data

    Returns:
        - If "leave_ET" stays 0, only all_data is returned.
        - If not, the function returns all_data, subsampled_data removing full echo trains from the end.
    '''
    # Determine dimensions
    header = ismrmrd.xsd.CreateFromDocument(dset.read_xml_header())
    ncoils = header.acquisitionSystemInformation.receiverChannels
    enc = header.encoding[0]
    if enc.encodingLimits.slice != None:
        nslices = enc.encodingLimits.slice.maximum + 1
    else:
        nslices = 1
    eNy = enc.encodedSpace.matrixSize.y
    eNx = enc.encodedSpace.matrixSize.x
    eNz = enc.encodedSpace.matrixSize.z

    # Initialize data arrays
    all_data = np.zeros((nslices, ncoils, eNz, eNy + 1, eNx), dtype=np.complex64)
    subsampled_data = np.zeros((nslices, ncoils, eNz, eNy + 1, eNx), dtype=np.complex64)

    # Check the noise samples to ignore
    firstacq = 0

    for acqnum in range(dset.number_of_acquisitions()):
        acq = dset.read_acquisition(acqnum)

        if acq.isFlagSet(ismrmrd.ACQ_IS_NOISE_MEASUREMENT):
            print("Found noise scan at acq ", acqnum)
            continue
        else:
            firstacq = acqnum
            print("Imaging acquisition starts acq ", acqnum)
            break

    for acqnum in range(firstacq, dset.number_of_acquisitions()):

        acq = dset.read_acquisition(acqnum)
        slice = acq.idx.slice
        y = acq.idx.kspace_encode_step_1
        z = acq.idx.kspace_encode_step_2 - 1
        set = acq._head.idx.set
        av = acq._head.idx.average + 1

        if eNz > 1: #This will be the case for the first dataset
            if set == 0 and z > 0 and av == 1:
                all_data[0, :, z, y, :] = acq.data
        else:
            if av == 1 or av == 2:  # only getting the first two averages
                all_data[slice, :,0, y, :] = acq.data

            if (acqnum <= etl * nslices * (1 + etc * 2 - leave_ET)):
                # Subsample data based on echo trains
                subsampled_data[slice, :,0 , y, :] = acq.data

    # If leave_ET is 0, return only all_data; otherwise, return both arrays
    if leave_ET == 0:
        return all_data
    else:
        return all_data, subsampled_data


if __name__ == "__main__":
    pass