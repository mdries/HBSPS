"""spectra module.

This module contains classes and functions related
to dealing with spectra

"""
import numpy as np
import scipy
from scipy import ndimage
from scipy.special import legendre

def smoothSpectrum(wavelength, spectrum, sigma):
	"""Smooth spectrum to a given velocity dispersion.
	
	Args:
		wavelength: wavelength-array of the spectrum (should
			be logarithmic for constant sigma-smoothing).
		spectrum: numpy array with spectral data.
		sigma: required velocity dispersion (km/s)

	Returns:
		spectrumSmooth: smoothed version of the spectrum.

	"""
	
	clight = 299792.458
	cdelt = np.log(wavelength[1]) - np.log(wavelength[0])
	sigmaPixel = sigma / (clight * cdelt)
	smoothSpectrum = smoothSpectrumFast(spectrum, sigmaPixel)
	
	return smoothSpectrum
	
def smoothSpectra(wavelength, S, sigma):
	"""Smooth spectra in matrix with stellar spectra to a given velocity dispersion.
	
	Args:
		wavelength: wavelength-array of the spectra (should 
			be logarithmic for constant sigma smoothing).
		S: matrix with stellar templates, spectra are assumed to be 
			int the columns of the matrix.
		spectrum: numpy array with spectral data.
		sigma: required velocity dispersion (km/s)

	Returns:
		S: smoothed version of the spectra in S.

	"""
	clight = 299792.458
	cdelt = np.log(wavelength[1]) - np.log(wavelength[0])
	sigmaPixel = sigma / (clight * cdelt)
	
	nTemplates = S.shape[1]
	for tIdx in range(nTemplates):
		S[:,tIdx] = smoothSpectrumFast(S[:,tIdx], sigmaPixel)
	
	return S
	

def smoothSpectrumFast(spectrum, sigmaPixel):
	"""Fast spectrum smoothing.
	
	This function smooths a spectrum given the
	standard deviation in pixel space.
	
	Args:
		spectrum: the input spectrum.
		sigmaPixel: smoothing scale in pixel space.
			
	Returns:
		smoothSpectrum: a smoothed version of the 
			input spectrum.
	
	"""
	
	smoothSpectrum = scipy.ndimage.gaussian_filter(spectrum, sigma=(sigmaPixel), order=0)
	
	return smoothSpectrum
	
def getGaussianLP(w, wc, wstd, norm):
	"""Calculate Gaussian line profile for local covariance structure
	
	"""
	glp = norm * np.exp(-((w-wc)**2/(2*wstd**2)))
	
	return glp
	
def getLegendrePolynomial(wavelength, order, bounds=None):
	nBins = len(wavelength)
	if bounds == None:
		wavelengthN = -1.0+2.0*(wavelength-wavelength[0])/(wavelength[-1]-wavelength[0])
	else:
		wavelengthN = -1.0+2.0*(wavelength-bounds[0])/(bounds[1]-bounds[0])
	
	AL = np.zeros([nBins,order+1])
	for oIdx in range(order+1):
		pDegree = oIdx
		legendreP = np.array(legendre(pDegree))
		for dIdx in range(pDegree+1):
			AL[:,pDegree] += legendreP[dIdx]*wavelengthN**(pDegree-dIdx)
				
	return AL
	
def rebin(x1, x2, y):
	"""Rebin a spectrum from grid x1 to grid x2
	
	Use this function to rebin a spectrum from grid x1 to grid
	x2. This routine conserves flux density but not flux.
	
	Args:
		x1: wavelength grid on which the spectrum is defined.
		x2: wavelength grid to which spectrum should be 
		rebinned.
		y: the spectrum (i.e. flux vector).
	
	"""
	
	# Define number of pixels in the two wavelength grids
	nPix1 = len(x1)
	nPix2 = len(x2)
	
	# Define the boundaries of the pixels for the
	# two wavelength grids
	step1 = x1[1:] - x1[:-1]
	step2 = x2[1:]-x2[:-1]
		
	binB1 = np.zeros(len(x1)+1)
	binB2 = np.zeros(len(x2)+1)
	
	binB1[0] = x1[0]-step1[0]/2.0
	binB1[1:-1] = x1[:-1] + step1/2.0
	binB1[-1] = x1[-1]+step1[-1]/2.0
	
	binB2[0] = x2[0]-step2[0]/2.0
	binB2[1:-1] = x2[:-1] + step2/2.0
	binB2[-1] = x2[-1]+step2[-1]/2.0
	
	# Determine where to insert boundaries of original
	# array into boundaries of the new array
	x2p = np.searchsorted(binB1, binB2)
	
	# Define rebinned flux vector
	b = np.zeros(len(x2))
	
	# Process all pixels of new wavelength grid
	# and find corresponding pixels in original 
	# wavelength grid that contribute to the flux 
	# in these pixels
	for pix in range(len(x2)):
		idS = max(x2p[pix]-1,0)
		idE = min((x2p[pix+1],len(x1)))
		for id1 in range(idS,idE):
			wL = max(binB1[id1],binB2[pix])
			wR = min(binB1[id1+1],binB2[pix+1])
			if id1 == 0:
				wL = binB2[pix]
			if id1 == len(x1)-1:
				wR = binB2[pix+1]
			b[pix] += y[id1] * (wR-wL) / (binB1[id1+1]-binB1[id1]) * (binB1[id1+1]-binB1[id1])
		b[pix] /= (binB2[pix+1]-binB2[pix])
		
	return b