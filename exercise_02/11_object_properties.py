import sys
import math
import cv2 as cv
import numpy as np

# Contours smaller than this area (in pixels) are treated as noise
MIN_AREA = 100

# Minimum difference from the estimated background to count as an object (0-255)
DIFF_THRESHOLD = 20

# Size of the median filter used to estimate the background.
# Must be odd and larger than the biggest object in the image.
BG_KERNEL = 151


def main(argv):
    # Path to the input image (adjust the name or use a full path)
    filename = "stuff.jpg"

    # Load the image
    src = cv.imread(filename, cv.IMREAD_COLOR)
    if src is None:
        print("Error opening image: " + filename)
        return -1

    # Convert to grayscale and smooth a little to reduce noise
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (5, 5), 0)

    # Estimate the (uneven) background with a large median filter.
    # Objects are smaller than the kernel, so they disappear from the estimate.
    background = cv.medianBlur(blurred, BG_KERNEL)

    # Objects = pixels that differ from the background, both darker and brighter
    diff = cv.absdiff(blurred, background)
    _, bw = cv.threshold(diff, DIFF_THRESHOLD, 255, cv.THRESH_BINARY)

    # Remove small specks, then close small holes inside the objects
    open_kernel = np.ones((3, 3), np.uint8)
    close_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (9, 9))
    bw = cv.morphologyEx(bw, cv.MORPH_OPEN, open_kernel)
    bw = cv.morphologyEx(bw, cv.MORPH_CLOSE, close_kernel)

    # Find the outer contours of the objects
    contours, _ = cv.findContours(bw, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv.contourArea(c) >= MIN_AREA]

    print("Objects found: " + str(len(contours)))
    print()
    header = f"{'ID':>3} {'cx':>8} {'cy':>8} {'AspectRatio':>12} {'Extent':>8} {'Solidity':>9} {'EqDiameter':>11} {'Orientation':>12} {'MeanInt':>8}"
    print(header)
    print("-" * len(header))

    result = src.copy()

    for i, cnt in enumerate(contours, start=1):
        area = cv.contourArea(cnt)

        # Center (centroid) from image moments
        m = cv.moments(cnt)
        if m["m00"] == 0:
            continue
        cx = m["m10"] / m["m00"]
        cy = m["m01"] / m["m00"]

        # Aspect ratio = width / height of the bounding rectangle
        x, y, w, h = cv.boundingRect(cnt)
        aspect_ratio = float(w) / h

        # Extent = object area / bounding rectangle area
        extent = area / float(w * h)

        # Solidity = object area / convex hull area
        hull = cv.convexHull(cnt)
        hull_area = cv.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0.0

        # Equivalent diameter = diameter of a circle with the same area
        equi_diameter = math.sqrt(4 * area / math.pi)

        # Orientation = angle of the fitted ellipse (needs at least 5 points)
        if len(cnt) >= 5:
            (_, _), (_, _), angle = cv.fitEllipse(cnt)
        else:
            angle = float("nan")

        # Mean intensity inside the object, measured on the grayscale image
        mask = np.zeros(gray.shape, np.uint8)
        cv.drawContours(mask, [cnt], -1, 255, -1)
        mean_intensity = cv.mean(gray, mask=mask)[0]

        print(f"{i:>3} {cx:>8.1f} {cy:>8.1f} {aspect_ratio:>12.3f} {extent:>8.3f} {solidity:>9.3f} {equi_diameter:>11.2f} {angle:>12.2f} {mean_intensity:>8.2f}")

        # Visualization
        cv.drawContours(result, [cnt], -1, (0, 255, 0), 2)
        cv.circle(result, (int(cx), int(cy)), 4, (0, 0, 255), -1)
        cv.putText(result, str(i), (int(cx) + 6, int(cy) - 6), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    cv.imshow("Source", src)
    cv.imshow("Difference from background", diff)
    cv.imshow("Binary mask", bw)
    cv.imshow("Detected objects", result)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return 0


if __name__ == "__main__":
    main(sys.argv[1:])