"""Desktop version of the recognizer, built with pygame.

Draw a digit with the mouse; the prediction updates after every stroke.
Keys: C / Backspace = clear, Esc = quit.
"""

from __future__ import annotations

import math

import numpy as np
import pygame
from PIL import Image

from .config import MODEL_PATH
from .model import DigitClassifier
from .preprocessing import preprocess

CANVAS_SIZE = 420
MARGIN = 24
PANEL_WIDTH = 250
WINDOW_SIZE = (MARGIN + CANVAS_SIZE + MARGIN + PANEL_WIDTH + MARGIN, MARGIN + CANVAS_SIZE + MARGIN)
BRUSH_RADIUS = round(CANVAS_SIZE * 0.065 / 2)  # same stroke thickness as the web canvas

WINDOW_BG = (23, 26, 33)
CANVAS_BG = (0, 0, 0)
INK = (255, 255, 255)
TEXT = (232, 234, 240)
MUTED = (154, 163, 181)
ACCENT = (141, 151, 255)
TRACK = (43, 48, 59)


def surface_to_image(surface: pygame.Surface) -> Image.Image:
    return Image.frombytes("RGB", surface.get_size(), pygame.image.tobytes(surface, "RGB"))


def draw_stroke(surface: pygame.Surface, start: tuple[int, int], end: tuple[int, int]) -> None:
    """Stamp round brush dabs along the segment, so strokes are equally thick in every direction."""
    steps = max(1, math.ceil(math.dist(start, end) / (BRUSH_RADIUS / 3)))
    for i in range(steps + 1):
        t = i / steps
        point = (start[0] + (end[0] - start[0]) * t, start[1] + (end[1] - start[1]) * t)
        pygame.draw.circle(surface, INK, point, BRUSH_RADIUS)


class DesktopApp:
    def __init__(self, classifier: DigitClassifier):
        self.classifier = classifier
        self.screen = pygame.display.set_mode(WINDOW_SIZE)
        pygame.display.set_caption("Handwritten Digit Recognizer")
        self.canvas = pygame.Surface((CANVAS_SIZE, CANVAS_SIZE))
        self.canvas_rect = self.canvas.get_rect(topleft=(MARGIN, MARGIN))
        self.fonts = {
            "digit": pygame.font.Font(None, 180),
            "title": pygame.font.Font(None, 30),
            "text": pygame.font.Font(None, 24),
        }
        self.last_point: tuple[int, int] | None = None
        self.probabilities: np.ndarray | None = None
        self.clear()

    def clear(self) -> None:
        self.canvas.fill(CANVAS_BG)
        self.probabilities = None

    def to_canvas(self, pos: tuple[int, int]) -> tuple[int, int]:
        return pos[0] - self.canvas_rect.x, pos[1] - self.canvas_rect.y

    def predict(self) -> None:
        digit_image = preprocess(surface_to_image(self.canvas))
        self.probabilities = None if digit_image is None else self.classifier.predict(digit_image)[0]

    def handle(self, event: pygame.event.Event) -> bool:
        """Process one event; return False when the app should quit."""
        if event.type == pygame.QUIT:
            return False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                return False
            if event.key in (pygame.K_c, pygame.K_n, pygame.K_BACKSPACE, pygame.K_DELETE):
                self.clear()
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and self.canvas_rect.collidepoint(event.pos):
            self.last_point = self.to_canvas(event.pos)
            pygame.draw.circle(self.canvas, INK, self.last_point, BRUSH_RADIUS)
        elif event.type == pygame.MOUSEMOTION and self.last_point is not None:
            point = self.to_canvas(event.pos)
            draw_stroke(self.canvas, self.last_point, point)
            self.last_point = point
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1 and self.last_point is not None:
            self.last_point = None
            self.predict()
        return True

    def draw_panel(self) -> None:
        x = self.canvas_rect.right + MARGIN
        y = MARGIN
        self.blit("Prediction", "title", TEXT, (x, y))

        if self.probabilities is None:
            digit, label = "?", "Draw a digit"
            digit_color = MUTED
        else:
            best = int(np.argmax(self.probabilities))
            digit, label = str(best), f"{self.probabilities[best]:.1%} sure"
            digit_color = ACCENT
        self.blit(digit, "digit", digit_color, (x, y + 34))
        self.blit(label, "text", MUTED, (x + 4, y + 160))

        bar_y = y + 196
        bar_width = PANEL_WIDTH - 70
        for d in range(10):
            p = 0.0 if self.probabilities is None else float(self.probabilities[d])
            is_top = self.probabilities is not None and d == int(np.argmax(self.probabilities))
            row_y = bar_y + d * 19
            self.blit(str(d), "text", ACCENT if is_top else TEXT, (x + 4, row_y))
            track = pygame.Rect(x + 24, row_y + 4, bar_width, 9)
            pygame.draw.rect(self.screen, TRACK, track, border_radius=4)
            fill_width = round(bar_width * p)
            if fill_width >= 2:
                fill = track.copy()
                fill.width = fill_width
                pygame.draw.rect(self.screen, ACCENT if is_top else MUTED, fill, border_radius=4)

        self.blit("C: clear   Esc: quit", "text", MUTED, (x + 4, self.canvas_rect.bottom - 18))

    def blit(self, text: str, font: str, color, pos) -> None:
        self.screen.blit(self.fonts[font].render(text, True, color), pos)

    def run(self) -> None:
        clock = pygame.time.Clock()
        while True:
            for event in pygame.event.get():
                if not self.handle(event):
                    return
            self.screen.fill(WINDOW_BG)
            self.screen.blit(self.canvas, self.canvas_rect)
            self.draw_panel()
            pygame.display.flip()
            clock.tick(120)


def run(model_path=MODEL_PATH) -> None:
    classifier = DigitClassifier(model_path)
    classifier.warm_up()  # fail fast with a clear message if the model is missing
    pygame.init()
    try:
        DesktopApp(classifier).run()
    finally:
        pygame.quit()
