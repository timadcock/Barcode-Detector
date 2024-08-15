import numpy as np
from numpy import linalg
from skimage import io, filters, feature, morphology, measure
from skimage.color import rgb2gray
import skimage.transform as skt
from scipy import ndimage, signal
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl
import os

F = False
T = True
from IPython.display import clear_output


# Gets all the images in a folder and converts them to grayscale
def getImageArray(folder):

    files = os.listdir(folder)
    images = []

    # since files will not be in the order I want it in I will set the images to the rigt position
    for f in files:
        tmp = io.imread(folder + "" + f)
        # print(f," : ",tmp.shape)
        images.append(tmp)

    return images


# Shows all the images in an array
def printImg(imgs):
    j = 1
    for i in imgs:
        plt.imshow(i, cmap="gray")
        plt.axis("off")
        p = "Image " + str(j)
        plt.title(p)
        plt.show()
        j += 1


# This will "bring out" the barcodes after the filter and thresholding, this is done by eroding the smaller white spots away then, dilate them afterwards.
def bringOut(img, iter=1):

    erode = morphology.binary_erosion(img)

    for j in range(iter - 1):
        erode = morphology.erosion(erode)

    dil = morphology.binary_dilation(erode)

    for j in range(iter - 1):
        dil = morphology.dilation(dil)

    return dil


# This will filter the image using the kernel in the frquency domain.
def fft(image, kernel):
    imgR, imgC = image.shape[:2]
    kernR, kernC = kernel.shape[:2]

    kernPad = np.zeros(image.shape[:2])

    sR = (imgR - kernR) // 2
    sC = (imgC - kernC) // 2

    kernPad[sR : sR + kernR, sC : sC + kernC] = kernel

    imgFFT = np.fft.fft2(image)
    kernFFT = np.fft.fft2(kernPad)

    filt = np.fft.fftshift(np.fft.ifft2(imgFFT * kernFFT))

    return filt / np.max(filt)


# This will get a gaussian kernel for use in filtering.
def getGauss(width, height, sigma):
    tmp = signal.get_window(("gaussian", sigma), width)
    tmp2 = signal.get_window(("gaussian", sigma), height)
    gauss = np.outer(tmp, tmp2)

    return gauss


# This will do a highpass filtering using the fft.
def highPass(img, kernel):
    filt = img / np.max(img) - fft(img, kernel)
    return filt


# Get the largest contour by area
def largestContour(conts):

    max = 0
    ind = 0
    # Using the SHoelace formula to calculate area of the contour
    for c in range(len(conts)):
        corners = conts[c]
        n = len(corners)  # of corners
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += corners[i][0] * corners[j][1]
            area -= corners[j][0] * corners[i][1]
        area = abs(area) / 2.0

        if max < area:
            max = area
            ind = c

    return ind


def SobelDetection(img, blur=3, bringout=3, v=True):
    cimg = img
    img = rgb2gray(img)

    # Some images are more
    if v:
        grad = filters.sobel_v(img)
    else:
        grad = filters.sobel(img)

    tmp = grad < 0

    grad[tmp] = 0

    gauss = filters.gaussian(grad, blur)

    fig, ax = filters.try_all_threshold(gauss)
    fig.show()
    plt.show()

    clear_output()

    plt.imshow(grad, cmap="gray")
    plt.axis("off")
    plt.show()

    plt.imshow(gauss, cmap="gray")
    plt.axis("off")
    plt.show()

    tmp = filters.threshold_yen(gauss)

    tmp1 = gauss
    th = gauss >= tmp

    tmp1[th] = 1

    th = gauss < tmp

    tmp1[th] = 0

    dil = bringOut(tmp1, bringout)

    plt.imshow(dil, cmap="gray")
    plt.axis("off")
    plt.show()

    cont = measure.find_contours(dil)

    # Display the image and plot all contours found
    fig, ax = plt.subplots()
    ax.imshow(cimg)

    ind = largestContour(cont)

    for n, cont in enumerate(cont):
        if n == ind:
            ax.plot(cont[:, 1], cont[:, 0], linewidth=4, color="r")

    ax.axis("image")
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()
