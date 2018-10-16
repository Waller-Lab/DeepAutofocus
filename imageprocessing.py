import numpy as np
from PIL import Image

def exporttiffstack(datacube, path="/Users/henrypinkard/Desktop/export.tif"):
    '''
    Save 3D numpy array as a TIFF stack
    :param datacube:
    '''
    if len(datacube.shape) == 2:
        imlist = [Image.fromarray(datacube)]
    else:
        imlist = []
        for i in range(datacube.shape[0]):
            imlist.append(Image.fromarray(datacube[i,...]))
    imlist[0].save(path, compression="tiff_deflate", save_all=True, append_images=imlist[1:])

def phasecorrelation(src_image, target_image):
    '''
    Compute rigid translational offset between two images
    :return:
    '''
    shape = src_image.shape
    src_ft = np.fft.fftn(src_image)
    target_ft = np.fft.fftn(target_image)
    normalized_cross_power_spectrum = (src_ft * target_ft.conj()) / np.abs(src_ft * target_ft.conj())
    normalized_cross_corr = np.fft.ifftn(normalized_cross_power_spectrum)
    # unnormalized_cross_corr = np.fft.ifftn((src_ft * target_ft.conj()))
    maxima = np.array(np.unravel_index(np.argmax(np.abs(normalized_cross_corr)),  normalized_cross_corr.shape))
    midpoints = np.array([np.fix(axis_size / 2) for axis_size in shape])
    shifts = np.array(maxima)
    shifts[shifts > midpoints] += np.array(shape)[shifts > midpoints] // 2
    shifts = np.mod(shifts, shape)
    return shifts

def computeregistrations(images, refchannel):
    '''
    compute pairwise registration shifts,
    :param images: dictionary with channel names as keys and pixels as values
    :return: dictionary with channel idex tuple as keys and shifts as value (e.g. (0,1))
    '''
    pairwiseshifts = {}
    for i, srcchannelname in enumerate(images.keys()):
        if srcchannelname == refchannel:
            pairwiseshifts[(refchannel, srcchannelname)] = np.zeros(next(iter(images.values())).ndim)
        else:
            pairwiseshifts[(refchannel, srcchannelname)] = phasecorrelation(images[refchannel], images[srcchannelname])
    return pairwiseshifts

def registerimages(images, shifts, refchannel):
    '''
    Register all images
    :param refchannel: dictionary with channel names as keys and pixels as values
    :param shifts:
    :param images:
    :return: all images cropped onto their common area of overlap, registered appropriately
    '''
    #find min and max shifts in order to get cropped image size
    shiftkeys = [key for key in shifts.keys() if key[0] == refchannel]
    shiftvals = np.array([shifts[key] for key in shiftkeys])
    #min and max are guaranteed to have 0s becuase they include reference channels shift with self
    #so maxshift guartenned to be >= 0
    minshift = np.min(shiftvals, axis=0)
    maxshift = np.max(shiftvals, axis=0)
    oldsize = images[refchannel].shape
    registered = {}
    for channel in images.keys():
        if channel == refchannel: #no shift for ref channel
            registered[channel] = images[channel][maxshift[0]: oldsize[0] + minshift[0], maxshift[1]: oldsize[1] + minshift[1]]
        else:
            shift = shifts[refchannel, channel]
            registered[channel] = images[channel][maxshift[0] - shift[0]:oldsize[0] - shift[0] + minshift[0],
                                  maxshift[1] - shift[1]:oldsize[1] - shift[1] + minshift[1]]
    return registered

def radialaverage(image):
    '''
    Radially average a square image
    :param image:
    :return:
    '''
    center = (image.shape[0] / 2, image.shape[1] / 2)
    #index pixels based on which bin they fall into
    dx, dy = np.meshgrid(np.arange(image.shape[0]) - center[0], np.arange(image.shape[1]) - center[1])
    dist = np.sqrt(dx ** 2 + dy ** 2)
    integerdist = np.round(dist).astype(np.int)
    maxdist = min(integerdist[:,-1]) #bottom edge is closer to center pixel than top
    radialavg = np.zeros(maxdist)
    for i in np.arange(maxdist):
        radialavg[i] = np.mean(image[integerdist==i])
    return radialavg

def autocorrelate(pixels):
    '''
    Fast autocorrelation using FFT
    :param pixels:
    :return:
    '''
    pixelsft = np.fft.fft2(pixels)
    powerspectrum = pixelsft * pixelsft.conj()
    autocorr = np.fft.fftshift(np.fft.ifft2(powerspectrum))
    return autocorr

def radialllyaveragedpowerspectrum(pixels):
    '''
    Compute radially averaged power spectrum magnitude
    :param pixels:
    :return:
    '''
    pixelsft = np.fft.fftshift(np.fft.fft2(pixels))
    powerspectrum = pixelsft * pixelsft.conj()
    return radialaverage(np.abs(powerspectrum))