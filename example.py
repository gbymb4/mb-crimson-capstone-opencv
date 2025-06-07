# -*- coding: utf-8 -*-
"""
Created on Sat Jun  7 18:34:12 2025

@author: Gavin
"""

import cv2

import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load the image
image = cv2.imread('ball_and_sun.png')
original = image.copy()

# Step 2: Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Step 3: Smooth the image to reduce noise
blurred = cv2.GaussianBlur(gray, (9, 9), 2)

# Step 4: Threshold to detect and remove the sun (very bright regions)
_, sun_mask = cv2.threshold(blurred, 240, 255, cv2.THRESH_BINARY)
inverse_sun_mask = 255 - sun_mask  # NumPy equivalent of cv2.bitwise_not

# Step 5: Mask the bright area out (NumPy equivalent of cv2.bitwise_and)
sun_filtered = blurred * (inverse_sun_mask // 255)

# Step 6: Detect circles (golf ball)
circles = cv2.HoughCircles(
    sun_filtered,
    cv2.HOUGH_GRADIENT,
    dp=1.2,
    minDist=100,
    param1=50,
    param2=30,
    minRadius=10,
    maxRadius=100
)

# Step 7: Draw detected circles
if circles is not None:
    circles = np.round(circles[0, :]).astype("int")
    for (x, y, r) in circles:
        # Draw the outer circle
        cv2.circle(original, (x, y), r, (0, 255, 0), 4)
        # Draw the center of the circle
        cv2.circle(original, (x, y), 2, (0, 0, 255), 3)
else:
    print("No circles detected.")

# Step 8: Display result
original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

plt.imshow(original_rgb)
plt.title("Golf Ball Detection")
plt.axis('off')
plt.show()


