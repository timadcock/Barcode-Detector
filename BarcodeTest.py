from BarcodeDetector import *

imgs = getImageArray("./Data/Barcodes/")
# printImg(imgs)


SobelDetection(imgs[2], 4, 10)
