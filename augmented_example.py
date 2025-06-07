# -*- coding: utf-8 -*-
"""
Created on Sat Jun  7 18:21:15 2025

@author: Gavin
"""

import cv2
import random

import numpy as np
import matplotlib.pyplot as plt

seed = 0
random.seed(seed)
np.random.seed(seed)

base_image = cv2.imread('ball_and_sun.png')
num_tries = 6

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i in range(num_tries):
    image = base_image.copy()

    brightness_factor = random.uniform(0.7, 1.3)
    image = np.clip(image * brightness_factor, 0, 255).astype(np.uint8)

    if random.random() < 0.5:
        k = random.choice([3, 5, 7])
        image = cv2.GaussianBlur(image, (k, k), 0)

    rows, cols = image.shape[:2]
    angle = random.uniform(-10, 10)
    scale = random.uniform(0.9, 1.1)
    M_rot = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, scale)
    image = cv2.warpAffine(image, M_rot, (cols, rows))

    tx, ty = random.randint(-10, 10), random.randint(-10, 10)
    M_trans = np.float32([[1, 0, tx], [0, 1, ty]])
    image = cv2.warpAffine(image, M_trans, (cols, rows))

    original = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    _, sun_mask = cv2.threshold(blurred, 240, 255, cv2.THRESH_BINARY)
    inverse_sun_mask = 255 - sun_mask
    sun_filtered = blurred * (inverse_sun_mask // 255)

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

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            cv2.circle(original, (x, y), r, (0, 255, 0), 4)
            cv2.circle(original, (x, y), 2, (0, 0, 255), 3)

    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    
    axes[i].imshow(original_rgb)
    axes[i].set_title(f"Try {i+1}")
    axes[i].axis('off')

plt.tight_layout()
plt.show()


