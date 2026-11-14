import sys
import math
import cv2 as cv
import numpy as np


def main(argv):
    default_file = "sudoku.png"
    filename = argv[0] if len(argv) > 0 else default_file

    # Load the image in grayscale
    src = cv.imread(cv.samples.findFile(filename), cv.IMREAD_GRAYSCALE)

    # Check if the image is loaded fine
    if src is None:
        print("Error opening image!")
        print("Usage: hough_lines.py [image_name -- default " + default_file + "] \n")
        return -1

    dst = cv.Canny(src, 50, 200, None, 3)

    # Copy edges to images that will display the results in BGR
    cdst = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)
    cdstP = np.copy(cdst)

    # Standard Hough Line Transform
    lines = cv.HoughLines(dst, 1, np.pi / 180, 150)

    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
            pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
            cv.line(cdst, pt1, pt2, (0, 0, 255), 3, cv.LINE_AA)

    # Probabilistic Hough Line Transform
    linesP = cv.HoughLinesP(dst, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=10)

    if linesP is not None:
        # reshape(-1, 4) works for both (N, 1, 4) and (N, 4) shapes
        for x1, y1, x2, y2 in linesP.reshape(-1, 4):
            cv.line(cdstP, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3, cv.LINE_AA)

    cv.imshow("Source", src)
    cv.imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst)
    cv.imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP)

    cv.waitKey()
    return 0


if __name__ == "__main__":
    main(sys.argv[1:])