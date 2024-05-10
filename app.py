import pygame
import sys
import numpy as np
from pygame.locals import *
from keras.models import load_model
import cv2

WINDOWSIZEX = 640
WINDOWSIZEY = 480
BOUNDARY = 5
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)

IMAGESAVE = False

# Path to the model file "bestmodel.h5"
MODEL = load_model(r"bestmodel.keras")

LABELS = {
    0: "Zero",
    1: "One",
    2: "Two",
    3: "Three",
    4: "Four",
    5: "Five",
    6: "Six",
    7: "Seven",
    8: "Eight",
    9: "Nine"
}

img_count = 1

# Initialize pygame
pygame.init()

# Load font
FONT = pygame.font.Font("freesansbold.ttf", 18)

# Set up display
DISPLAYSURF = pygame.display.set_mode((WINDOWSIZEX, WINDOWSIZEY))
pygame.display.set_caption("Number Detector")

iswriting = False
number_xcord = []
number_ycord = []
Predict = True

while True:
    # Event loop
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()

        # Handle mouse events
        if event.type == MOUSEMOTION and iswriting:
            xcord, ycord = event.pos
            pygame.draw.circle(DISPLAYSURF, WHITE, (xcord, ycord), 4, 0)
            number_xcord.append(xcord)
            number_ycord.append(ycord)

        if event.type == MOUSEBUTTONDOWN:
            iswriting = True

        if event.type == MOUSEBUTTONUP:
            iswriting = False
            number_xcord = sorted(number_xcord)
            number_ycord = sorted(number_ycord)
            rect_min_x, rec_max_x = max(number_xcord[0]-BOUNDARY, 0), min(WINDOWSIZEX, number_xcord[-1]+BOUNDARY)
            rect_min_y, rec_max_y = max(number_ycord[0]-BOUNDARY, 0), min(number_ycord[-1]+BOUNDARY, WINDOWSIZEY)
            number_xcord = []
            number_ycord = []
            img_arr = np.array(pygame.PixelArray(DISPLAYSURF))[rect_min_x:rec_max_x, rect_min_y:rec_max_y].T.astype(np.float32)

            if IMAGESAVE:
                cv2.imwrite(f"image{img_count}.png", img_arr)
                img_count += 1

            if Predict:
                image = cv2.resize(img_arr, (28, 28))
                image = np.pad(image, (10, 10), 'constant', constant_values=0)
                image = cv2.resize(image, (28, 28)) / 255
                label = str(LABELS[np.argmax(MODEL.predict(image.reshape(1, 28, 28, 1)))])
                textSurface = FONT.render(label, True, RED, WHITE)
                textRecObj = textSurface.get_rect()
                textRecObj.left, textRecObj.bottom = rect_min_x, rec_max_y
                DISPLAYSURF.blit(textSurface, textRecObj)

        # Handle key events
        if event.type == KEYDOWN:
            if event.unicode == "n":
                DISPLAYSURF.fill(BLACK)

    # Update display
    pygame.display.update()

