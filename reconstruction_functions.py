### IMPORTS
from sewar.full_ref import filter2, fspecial, Filter, uniform_filter
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
import warnings
from scipy import ndimage
from scipy.fft import fftn, ifftn, fftshift, ifftshift
from typing import *
from numpy.typing import ArrayLike
from matplotlib.axes import Axes




### SECTION 1

def _initial_check(GT,P):
    assert GT.shape == P.shape, "Supplied images have different sizes " + \
    str(GT.shape) + " and " + str(P.shape)
    if GT.dtype != P.dtype:
        msg = "Supplied images have different dtypes " + \
            str(GT.dtype) + " and " + str(P.dtype)
        warnings.warn(msg)

    if len(GT.shape) == 2:
        GT = GT[:,:,np.newaxis]
        P = P[:,:,np.newaxis]

    return GT.astype(np.float64),P.astype(np.float64)


def _get_sums(GT,P,win,mode='same'):
    mu1,mu2 = (filter2(GT,win,mode),filter2(P,win,mode))
    return mu1*mu1, mu2*mu2, mu1*mu2


def _get_sigmas(GT,P,win,mode='same',**kwargs):
    if 'sums' in kwargs:
        GT_sum_sq,P_sum_sq,GT_P_sum_mul = kwargs['sums']
    else:
        GT_sum_sq,P_sum_sq,GT_P_sum_mul = _get_sums(GT,P,win,mode)

    return filter2(GT*GT,win,mode)  - GT_sum_sq,\
            filter2(P*P,win,mode)  - P_sum_sq, \
            filter2(GT*P,win,mode) - GT_P_sum_mul


def _ssim_single (GT,P,ws,C1,C2,fltr_specs,mode):
    win = fspecial(**fltr_specs)

    GT_sum_sq,P_sum_sq,GT_P_sum_mul = _get_sums(GT,P,win,mode)
    sigmaGT_sq,sigmaP_sq,sigmaGT_P = _get_sigmas(GT,P,win,mode,sums=(GT_sum_sq,P_sum_sq,GT_P_sum_mul))

    assert C1 > 0
    assert C2 > 0

    ssim_map = ((2*GT_P_sum_mul + C1)*(2*sigmaGT_P + C2))/((GT_sum_sq + P_sum_sq + C1)*(sigmaGT_sq + sigmaP_sq + C2))
    cs_map = (2*sigmaGT_P + C2)/(sigmaGT_sq + sigmaP_sq + C2)
    return np.mean(ssim_map), np.mean(cs_map)


def ssim(gt: np.ndarray, p: np.ndarray, ws: int = 11, k1: float = 0.01, k2: float = 0.03,
         max_val: Optional[int] = None, fltr_specs: Optional[Dict[str, int]] = None,
         mode: str = 'valid') -> Tuple[float, float]:
    """
    Calculates structural similarity index (ssim).

    Parameters:
    - gt (np.ndarray): The first (original) input image.
    - p (np.ndarray): The second (deformed) input image.
    - ws (int): Sliding window size (default = 11).
    - k1 (float): First constant for SSIM (default = 0.01).
    - k2 (float): Second constant for SSIM (default = 0.03).
    - max_val (Optional[int]): Maximum value of datarange (if None, max_val is calculated using image dtype).
    - fltr_specs (Optional[Dict[str, int]]): Filter specifications (default = None).
    - mode (str): Convolution mode for valid (default = 'valid').

    Returns:
    - Tuple[float, float]: ssim value, cs value.
    """
    if max_val is None:
        max_val = np.iinfo(gt.dtype).max

    gt, p = _initial_check(gt, p)

    if fltr_specs is None:
        fltr_specs = dict(fltr=Filter.UNIFORM, ws=ws)

    c1 = (k1 * max_val)**2
    c2 = (k2 * max_val)**2

    ssims = []
    css = []
    for i in range(gt.shape[2]):
        ssim_val, cs = _ssim_single(gt[:, :, i], p[:, :, i], ws, c1, c2, fltr_specs, mode)
        ssims.append(ssim_val)
        css.append(cs)
    return np.mean(ssims), np.mean(css)


def mse(gt: np.ndarray, p: np.ndarray) -> float:
    """
    Calculates mean squared error (mse) between two images.

    Parameters:
    - gt (np.ndarray): The first (original) input image.
    - p (np.ndarray): The second (deformed) input image.

    Returns:
    - float: mse value.
    """
    return np.mean((gt.astype(np.float64) - p.astype(np.float64))**2)


def rmse(gt: np.ndarray, p: np.ndarray) -> float:
    """
    Calculates root mean squared error (rmse).

    Parameters:
    - gt (np.ndarray): The first (original) input image.
    - p (np.ndarray): The second (deformed) input image.

    Returns:
    - float: rmse value.
    """
    gt, p = _initial_check(gt, p)
    return np.sqrt(mse(gt, p))


def _vifp_single(GT,P,sigma_nsq):
    EPS = 1e-10
    num =0.0
    den =0.0
    for scale in range(1,5):
        N=2.0**(4-scale+1)+1
        win = fspecial(Filter.GAUSSIAN,ws=N,sigma=N/5)

        if scale >1:
            GT = filter2(GT,win,'valid')[::2, ::2]
            P = filter2(P,win,'valid')[::2, ::2]

        GT_sum_sq,P_sum_sq,GT_P_sum_mul = _get_sums(GT,P,win,mode='valid')
        sigmaGT_sq,sigmaP_sq,sigmaGT_P = _get_sigmas(GT,P,win,mode='valid',sums=(GT_sum_sq,P_sum_sq,GT_P_sum_mul))


        sigmaGT_sq[sigmaGT_sq<0]=0
        sigmaP_sq[sigmaP_sq<0]=0

        g=sigmaGT_P /(sigmaGT_sq+EPS)
        sv_sq=sigmaP_sq-g*sigmaGT_P

        g[sigmaGT_sq<EPS]=0
        sv_sq[sigmaGT_sq<EPS]=sigmaP_sq[sigmaGT_sq<EPS]
        sigmaGT_sq[sigmaGT_sq<EPS]=0

        g[sigmaP_sq<EPS]=0
        sv_sq[sigmaP_sq<EPS]=0

        sv_sq[g<0]=sigmaP_sq[g<0]
        g[g<0]=0
        sv_sq[sv_sq<=EPS]=EPS


        num += np.sum(np.log10(1.0+(g**2.)*sigmaGT_sq/(sv_sq+sigma_nsq)))
        den += np.sum(np.log10(1.0+sigmaGT_sq/sigma_nsq))

    return num/den


def vifp(gt: np.ndarray, p: np.ndarray, sigma_nsq: float = 2) -> float:
    """
    Calculates Pixel Based Visual Information Fidelity (vif-p).

    Parameters:
    - gt (np.ndarray): The first (original) input image.
    - p (np.ndarray): The second (deformed) input image.
    - sigma_nsq (float): Variance of the visual noise (default = 2).

    Returns:
    - float: vif-p value.
    """
    gt, p = _initial_check(gt, p)
    return np.mean([_vifp_single(gt[:, :, i], p[:, :, i], sigma_nsq) for i in range(gt.shape[2])])


def _power_complex(a,b):
        return a.astype('complex') ** b


def msssim(gt: np.ndarray, p: np.ndarray, weights: Union[List[float], np.ndarray] = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333],
           ws: int = 11, k1: float = 0.01, k2: float = 0.03, max_val: Optional[int] = None) -> float:
    """
    Calculates multi-scale structural similarity index (ms-ssim).

    Parameters:
    - gt (np.ndarray): The first (original) input image.
    - p (np.ndarray): The second (deformed) input image.
    - weights (Union[List[float], np.ndarray]): Weights for each scale (default = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).
    - ws (int): Sliding window size (default = 11).
    - k1 (float): First constant for SSIM (default = 0.01).
    - k2 (float): Second constant for SSIM (default = 0.03).
    - max_val (Optional[int]): Maximum value of datarange (if None, max_val is calculated using image dtype).

    Returns:
    - float: ms-ssim value.
    """
    if max_val is None:
        max_val = np.iinfo(gt.dtype).max

    gt, p = _initial_check(gt, p)

    scales = len(weights)

    fltr_specs = dict(fltr=Filter.GAUSSIAN, sigma=1.5, ws=11)

    if isinstance(weights, list):
        weights = np.array(weights)

    mssim_vals = []
    mcs_vals = []
    for _ in range(scales):
        _ssim, _cs = ssim(gt, p, ws=ws, k1=k1, k2=k2, max_val=max_val, fltr_specs=fltr_specs)
        mssim_vals.append(_ssim)
        mcs_vals.append(_cs)

        filtered = [uniform_filter(im, 2) for im in [gt, p]]
        gt, p = [x[::2, ::2, :] for x in filtered]

    mssim_vals = np.array(mssim_vals, dtype=np.float64)
    mcs_vals = np.array(mcs_vals, dtype=np.float64)

    return np.prod(_power_complex(mcs_vals[:scales - 1], weights[:scales - 1])) * _power_complex(mssim_vals[scales - 1], weights[scales - 1])


def echo_train_length(dset: ismrmrd.Dataset) -> Optional[int]:
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


def echo_train_count(dset: ismrmrd.Dataset) -> Optional[int]:
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


def normalize8(image: ArrayLike) -> np.ndarray:
    """
    Normalize an input image array to the range [0, 255] and convert to 8-bit unsigned integers.

    Parameters:
    - image (ArrayLike): Input image array containing numeric pixel values to be normalized.

    Returns:
    - numpy.ndarray: Normalized image array with pixel values in the range [0, 255] as 8-bit unsigned integers.
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


def iqm(im1: np.ndarray, im2: np.ndarray, string: bool = True) -> Union[str, tuple]:
    """
    Calculate and return various image quality metrics between two input images.

    Parameters:
    - im1 (np.ndarray): The first input image for comparison.
    - im2 (np.ndarray): The second input image for comparison.
    - string (bool): If True, return the metrics as a formatted string.
                     If False, return the metrics as a tuple of numeric values.

    Returns:
    - str or tuple: If string=True, a formatted string containing SSIM, VIFP, and MS-SSIM metrics.
                    If string=False, a tuple containing SSIM, VIFP, and MS-SSIM values.
    """
    # Calculate SSIM, RMSE, VIFP, MS-SSIM, and PSNR metrics
    ssim_score = ssim(im1, im2)
    vifp_score = vifp(im1, im2)
    msssim_score = msssim(im1, im2)

    if string:
        # Return metrics as a formatted string
        metrics_string = f'SSIM: {ssim_score}\nVIFP: {vifp_score}\nMS-SSIM: {msssim_score}'
        return metrics_string
    else:
        # Return metrics as a tuple of numeric values
        return ssim_score, vifp_score, msssim_score


def homogeneous_mask(data: ArrayLike, r: int) -> np.ndarray:
    """
    Create a homogeneous mask for a given 4D data array by setting values at regular intervals to zero.

    Parameters:
    - data (ArrayLike): Input 4D data array.
    - R (int): Interval for setting values to zero. For example, if R=2, every second value will be set to zero.

    Returns:
    - masked_data (numpy.ndarray): 4D data array with values set to zero at regular intervals defined by R.
    """
    masked_data = data.copy()
    # Create a mask with the same shape as the input data, initialized with zeros
    mask = np.zeros_like(data)

    # Set values to 1 in the mask at regular intervals along the specified dimension
    mask[:, :, :, ::r, :] = 1

    # Set values in the input data to zero where the mask is zero
    masked_data[mask == 0] = 0

    return masked_data


def zero_pad_zy(data: ArrayLike, scaling_factor: float = 2.0) -> np.ndarray:
    """
    Zero-pads the input data along the Z and Y dimensions.

    Parameters:
    - data (ArrayLike): The input data to be zero-padded. It's assumed to have at least 4 dimensions.
    - scaling_factor (float, optional): The scaling factor for the zero-padding. Default is 2.0.

    Returns:
    - padded_data (numpy.ndarray): The zero-padded data with modified dimensions according to the scaling factor.
    """

    # Get the original shape of the data along Z and Y dimensions
    sh_z, sh_y = [np.shape(data)[2], np.shape(data)[3]]

    # Calculate the new dimensions after applying the scaling factor
    im_z, im_y = int(scaling_factor * sh_z), int(scaling_factor * sh_y)

    # Calculate the starting index for zero-padding along Z and Y dimensions
    z_0 = im_z // 2 - (sh_z // 2)
    y_0 = im_y // 2 - (sh_y // 2)

    # Create a new array with modified dimensions for zero-padded data
    padded_shape = list(data.shape)
    padded_shape[2] = int(scaling_factor * padded_shape[2])
    padded_shape[3] = int(scaling_factor * padded_shape[3])
    padded_data = np.zeros(padded_shape, dtype=np.complex64)

    # Copy the original data into the zero-padded array at the appropriate indices
    padded_data[:, :, z_0:z_0 + sh_z, y_0:y_0 + sh_y, :] = data

    return padded_data


def image_order(data: np.ndarray) -> np.ndarray:
    """
    Rearrange the data array to a specific order by interleaving two halves of the data along a dimension.

    Parameters:
    - data (numpy.ndarray): Input data array to be rearranged.

    Returns:
    - ordered_data (numpy.ndarray): Rearranged data array with two halves interleaved along a dimension.
    """
    # Create a new complex array with the same shape as the input data, initialized with zeros
    ordered_data = np.zeros_like(data, dtype=np.complex64)

    # Calculate the number of rows for the first half of the data
    half = ordered_data[::2, :, :].shape[0]

    # Interleave the two halves of the data along the specified dimension
    ordered_data[::2] = data[:half]
    ordered_data[1::2] = data[half:]

    return ordered_data


def set_size(w: float, h: float, ax: Axes = None) -> None:
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


def imshow(image_matrix: np.ndarray, tile_shape: tuple = None, scale: tuple = None,
           titles: list = [], fontsize: int = 100, colorbar: bool = False, cmap: str = 'jet',
           size: list = [10, 10], text: str = '', image_name: str = '') -> None:
    """
    Tiles images and displays them in a window.

    Parameters:
    - image_matrix (np.ndarray): A 2D or 3D set of image data.
    - tile_shape (tuple, optional): Shape ``(rows, cols)`` for tiling images.
    - scale (tuple, optional): ``(min,max)`` values for scaling all images.
    - titles (list, optional): List of titles for each subplot.
    - fontsize (int): Font size for titles and text.
    - colorbar (bool): Whether to display colorbars for each image.
    - cmap (str): Colormap for all images.
    - size (list): Size of the figure in inches.
    - text (str): Additional text to be displayed in the figure.
    - image_name (str): Name of the saved image file (if provided).

    Notes:
    - Added some changes on matplotlib.pyplot.imshow, ref: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imshow.html
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


def crop_array(arr: np.ndarray, x: int, y: int, offx: int = 0, offy: int = 0) -> np.ndarray:

    """
    Crop the last two dimensions of a multidimensional array to the size of x,y.

    Parameters:
    - arr (np.ndarray): Numpy array of shape (..., H, W), where H and W are the height and width of the last two dimensions.
    - x (int): Integer representing the desired height of the cropped array.
    - y (int): Integer representing the desired width of the cropped array.
    - offx (int, optional): Integer representing the vertical offset (default is 0).
    - offy (int, optional): Integer representing the horizontal offset (default is 0).

    Returns:
    - cropped_arr (np.ndarray): Numpy array of shape (..., x, y), representing the cropped version of the input array.
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


def pad_image_stack(images: Union[list, np.ndarray]) -> np.ndarray:
    """
    Pad a stack of 2D images to a common size of 512x512.

    Parameters:
    - images Union[list, np.ndarray]: List or array of 2D image arrays.

    Returns:
    - padded_images (np.ndarray): Array of padded 2D image arrays with a common size of 512x512.
    """
    # Initialize an array to store padded images
    padded_images = np.zeros((len(images), 512, 512), dtype=np.complex64)

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


def grappa(kspace: np.ndarray, calib: np.ndarray, kernel_size: Tuple[int, int] = (5, 5), coil_axis: int = -1,
           lamda: float = 0.01, memmap: bool = False, memmap_filename: str = 'out.memmap', silent: bool = True
           ) -> Union[np.ndarray, None]:
    """
    GeneRalized Autocalibrating Partially Parallel Acquisitions.

    Parameters:
    - kspace (np.ndarray): 2D multi-coil k-space data to reconstruct from. Ensure that the missing entries have exact zeros.
    - calib (np.ndarray): Calibration data (fully sampled k-space).
    - kernel_size (Tuple[int, int], optional): Size of the 2D GRAPPA kernel (kx, ky).
    - coil_axis (int, optional): Dimension holding coil data. The other two dimensions should be image size: (sx, sy).
    - lamda (float, optional): Tikhonov regularization for the kernel calibration.
    - memmap (bool, optional): Store data in Numpy memmaps. Use when datasets are too large to store in memory.
    - memmap_filename (str, optional): Name of memmap to store results in. File is only saved if memmap=True.
    - silent (bool, optional): Suppress messages to the user.

    Returns:
    - res (np.ndarray or None): k-space data where missing entries have been filled in.

    -----
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
    """

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


def transform_kspace_to_image(k: np.ndarray, dim: Optional[Union[int, List[int]]] = None,
                              img_shape: Optional[Union[int, Tuple[int, ...]]] = None) -> np.ndarray:
    """
    Computes the Fourier transform from k-space to image space along given or all dimensions.

    Args:
    - k (np.ndarray): k-space data (complex-valued).
    - dim (Optional[Union[int, List[int]]]): Vector of dimensions to transform (default is all dimensions).
    - img_shape (Optional[Union[int, Tuple[int, ...]]]): Desired shape of the output image.

    Returns:
    - img (np.ndarray): Data in image space along transformed dimensions.
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


def ismrm_eig_power(r: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute dominant eigenvectors using the power method.

    Args:
    - r (np.ndarray): 4D array representing sample correlation matrices (rows x cols x ncoils x ncoils).

    Returns:
    - v (np.ndarray): Dominant eigenvectors corresponding to the sample correlation matrices.
    - d (np.ndarray): Dominant eigenvalues corresponding to the sample correlation matrices.
    """
    rows, cols, ncoils, _ = r.shape
    n_iterations = 2
    v = np.ones((rows, cols, ncoils), dtype=complex)  # Initialize eigenvectors

    d = np.zeros((rows, cols))
    for i in range(n_iterations):
        # Calculate the matrix-vector product r*v along the last dimension
        v = np.sum(r * np.tile(v[:, :, :, np.newaxis], (1, 1, 1, ncoils)), axis=2)

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


def ismrm_correlation_matrix(s: np.ndarray) -> np.ndarray:
    """
    Compute the sample correlation matrix for coil sensitivity maps.

    Args:
    - s (np.ndarray): 3D array representing coil sensitivity maps (rows x cols x ncoils).

    Returns:
    - rs (np.ndarray): 4D array representing the sample correlation matrix (rows x cols x ncoils x ncoils).
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


def smooth(img: np.ndarray, box: int = 5) -> np.ndarray:
    '''Smooths coil images

    :param img: Input complex images, ``[y, x] or [z, y, x]``
    :param box: Smoothing block size (default ``5``)

    :returns simg: Smoothed complex image ``[y,x] or [z,y,x]``
    '''

    t_real = np.zeros(img.shape, dtype=np.float64)
    t_imag = np.zeros(img.shape, dtype=np.float64)

    ndimage.filters.uniform_filter(img.real, size=box, output=t_real)
    ndimage.filters.uniform_filter(img.imag, size=box, output=t_imag)

    simg = t_real + 1j * t_imag

    return simg


def ismrm_estimate_csm_walsh(img: np.ndarray, smoothing: int = 5) -> np.ndarray:

    """
    Estimate coil sensitivity maps using the Walsh method.

    Args:
    - img: 3D array representing complex image data (height x width x ncoils).
    - smoothing: Integer representing the size of spatial smoothing window (optional, default is 5).

    Returns:
    - csm: Coil sensitivity maps estimated using the Walsh method.

    ----
    Note: Python conversion of the funtions from https://github.com/hansenms/ismrm_sunrise_matlab/blob/master/ismrm_est
    imate_csm_walsh.m Estimates relative coil sensitivity maps from a set of coil images using the eigenvector method
    described by Walsh et al. (Magn Reson Med2000;43:682-90.)
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


def calculate_sense_unmixing(acc_factor: float, csm: np.ndarray, regularization_factor: float = 0.001) -> Tuple[np.ndarray, np.ndarray]:

    '''
    Calculates the unmixing coefficients for a 2D image using a SENSE algorithm

    Args:
    - acc_factor: Acceleration factor, e.g. 2
    - csm: Coil sensitivity map, shape (coil, y, x)
    - regularization_factor: adds tychonov regularization (default 0.001)

    Returns:
    - unmix: Image unmixing coefficients for a single x location, shape (coil, y, x)
    - gmap: Noise enhancement map, shape (y, x)
    '''

    assert csm.ndim == 3, "Coil sensitivity map must have exactly 3 dimensions"

    unmix = np.zeros(csm.shape, np.complex64)

    for x in range(0, csm.shape[2]):
        unmix[:, :, x] = _calculate_sense_unmixing_1d(acc_factor, np.squeeze(csm[:, :, x]), regularization_factor)

    gmap = np.squeeze(np.sqrt(np.sum(abs(unmix) ** 2, 0))) * np.squeeze(np.sqrt(np.sum(abs(csm) ** 2, 0)))

    return unmix, gmap


def sample_data_ET_mask(dset: ismrmrd.Dataset, etl: int, etc: int, leave_ET: int = 0) -> np.ndarray:

    '''
    This functions samples the data into a numpy array with the option of subsampling it removing echo trains.

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