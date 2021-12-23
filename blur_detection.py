import pywt
import cv2
import numpy as np
import os
import argparse
import json


def blur_detect(img, threshold):
    # Convert to grayscale
    Y = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    M, N = Y.shape

    # Crop image to be 3 divisible by 2
    Y = Y[0:int(M / 16) * 16, 0:int(N / 16) * 16]

    # Compute Haar wavelet of image
    LL1, (LH1, HL1, HH1) = pywt.dwt2(Y, 'haar')
    # Another application of 2D haar to LL1
    LL2, (LH2, HL2, HH2) = pywt.dwt2(LL1, 'haar')
    # Another application of 2D haar to LL2
    LL3, (LH3, HL3, HH3) = pywt.dwt2(LL2, 'haar')

    # Construct the edge map in each scale Step 2
    E1 = np.sqrt(np.power(LH1, 2) + np.power(HL1, 2) + np.power(HH1, 2))
    E2 = np.sqrt(np.power(LH2, 2) + np.power(HL2, 2) + np.power(HH2, 2))
    E3 = np.sqrt(np.power(LH3, 2) + np.power(HL3, 2) + np.power(HH3, 2))

    M1, N1 = E1.shape

    # Sliding window size level 1
    sizeM1 = 8
    sizeN1 = 8

    # Sliding windows size level 2
    sizeM2 = int(sizeM1 / 2)
    sizeN2 = int(sizeN1 / 2)

    # Sliding windows size level 3
    sizeM3 = int(sizeM2 / 2)
    sizeN3 = int(sizeN2 / 2)

    # Number of edge maps, related to sliding windows size
    n_iter = int((M1 / sizeM1) * (N1 / sizeN1))

    Emax1 = np.zeros(n_iter)
    Emax2 = np.zeros(n_iter)
    Emax3 = np.zeros(n_iter)

    count = 0

    # Sliding windows index of level 1
    x1 = 0
    y1 = 0

    # Sliding windows index of level 2
    x2 = 0
    y2 = 0

    # Sliding windows index of level 3
    x3 = 0
    y3 = 0

    # Sliding windows limit on horizontal dimension
    y_limit = N1 - sizeN1

    while count < n_iter:
        # Get the maximum value of slicing windows over edge maps in each level
        Emax1[count] = np.max(E1[x1:x1 + sizeM1, y1:y1 + sizeN1])
        Emax2[count] = np.max(E2[x2:x2 + sizeM2, y2:y2 + sizeN2])
        Emax3[count] = np.max(E3[x3:x3 + sizeM3, y3:y3 + sizeN3])

        # If sliding windows ends horizontal direction move
        # along vertical direction and reset horizontal direction
        if y1 == y_limit:
            x1 = x1 + sizeM1
            y1 = 0

            x2 = x2 + sizeM2
            y2 = 0

            x3 = x3 + sizeM3
            y3 = 0

            count += 1

        else:
            y1 = y1 + sizeN1
            y2 = y2 + sizeN2
            y3 = y3 + sizeN3
            count += 1

    edge_point1 = Emax1 > threshold
    edge_point2 = Emax2 > threshold
    edge_point3 = Emax3 > threshold

    # Edge Points
    edge_point = edge_point1 + edge_point2 + edge_point3
    n_edges = edge_point.shape[0]

    # Dirak-Structure or Astep-Structure
    DAstructure = (Emax1[edge_point] > Emax2[edge_point]) * (Emax2[edge_point] > Emax3[edge_point])

    # Roof-Structure or Gstep-Structure
    RGstructure = np.zeros(n_edges)
    for i in range(n_edges):
        if edge_point[i] == 1:
            if Emax1[i] < Emax2[i] < Emax3[i]:
                RGstructure[i] = 1

    # Roof-Structure
    RSstructure = np.zeros(n_edges)
    for i in range(n_edges):
        if edge_point[i] == 1:
            if Emax2[i] > Emax1[i] and Emax2[i] > Emax3[i]:
                RSstructure[i] = 1

    # The edge is more likely to be in a blurred image
    BlurC = np.zeros(n_edges)
    for i in range(n_edges):
        if RGstructure[i] == 1 or RSstructure[i] == 1:
            if Emax1[i] < threshold:
                BlurC[i] = 1

    per = np.sum(DAstructure) / np.sum(edge_point)

    if (np.sum(RGstructure) + np.sum(RSstructure)) == 0:

        blur_extent = 100
    else:
        blur_extent = np.sum(BlurC) / (np.sum(RGstructure) + np.sum(RSstructure))

    return per, blur_extent


def images(input_dir):
    extensions = [".jpg", ".png", ".jpeg"]

    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if os.path.splitext(file)[1].lower() in extensions:
                yield os.path.join(root, file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Wavelet Blur Detector on a folder')
    parser.add_argument('-i', '--input_dir', dest="input_dir", type=str, required=True, help="directory of images")
    parser.add_argument('-s', '--save', dest='save_path', type=str, help="path to save output")
    parser.add_argument("-t", "--threshold", dest='threshold', type=float, default=35, help="blurry threshold")
    parser.add_argument("-d", "--decision", dest='MinZero', type=float, default=0.001, help="per Decision Threshold")
    args = parser.parse_args()

    # List for further data preservation
    results = []

    for input_path in images(args.input_dir):
        try:
            im = cv2.imread(input_path)
            p, blurext = blur_detect(im, args.threshold)
            if p < args.MinZero:
                classification = True
            else:
                classification = False
            results.append({"input_path": input_path, "per": p, "blur extent": blurext, "is blur": classification})
            print("{0}, Per: {1:.5f}, blur extent: {2:.3f}, is blur: {3}".format(input_path, p, blurext,
                                                                                 classification))
        except Exception as e:
            print(e)
            pass

    if args.save_path:
        assert os.path.splitext(args.save_path)[
                   1] == ".json", "The save path must be of .json extension"

        with open(args.save_path, 'w') as outfile:
            json.dump(results, outfile, indent=4)
