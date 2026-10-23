import numpy as np
import pytest
from PIL import Image

from digit_recognizer.preprocessing import preprocess


def centre_of_mass(image):
    ys, xs = np.indices(image.shape)
    return (ys * image).sum() / image.sum(), (xs * image).sum() / image.sum()


@pytest.mark.parametrize("background", [0, 255])
def test_blank_image_has_no_digit(background):
    assert preprocess(Image.new("L", (280, 280), background)) is None


def test_fully_transparent_image_has_no_digit():
    assert preprocess(Image.new("RGBA", (280, 280), (0, 0, 0, 0))) is None


def test_output_is_mnist_shaped(draw_seven):
    result = preprocess(draw_seven())
    assert result.shape == (28, 28)
    assert result.dtype == np.float32
    assert result.min() >= 0 and result.max() <= 1
    assert result.max() > 0.9


def test_digit_is_scaled_to_20px_and_centred_by_mass(draw_seven):
    # A small digit tucked into a corner, like a careless drawing.
    result = preprocess(draw_seven(left=10, top=10, size=60))
    rows, cols = np.nonzero(result > 0.1)
    assert max(rows.max() - rows.min(), cols.max() - cols.min()) + 1 <= 21
    cy, cx = centre_of_mass(result)
    assert abs(cy - 13.5) <= 1 and abs(cx - 13.5) <= 1


def test_position_and_size_on_the_canvas_do_not_matter(draw_seven):
    big_centred = preprocess(draw_seven(left=60, top=50, size=160))
    small_offset = preprocess(draw_seven(left=150, top=150, size=90))
    assert np.abs(big_centred - small_offset).mean() < 0.05


def test_dark_ink_on_light_paper_is_inverted(draw_seven):
    expected = preprocess(draw_seven(ink=255, background=0))
    result = preprocess(draw_seven(ink=0, background=255))
    np.testing.assert_allclose(result, expected, atol=0.02)


def test_greyish_photo_is_contrast_stretched(draw_seven):
    expected = preprocess(draw_seven(ink=255, background=0))
    result = preprocess(draw_seven(ink=(40, 40, 50), background=(200, 195, 190), mode="RGB"))
    np.testing.assert_allclose(result, expected, atol=0.05)


@pytest.mark.parametrize("ink", [(0, 0, 0, 255), (255, 255, 255, 255)])
def test_transparent_background_works_with_dark_or_light_ink(draw_seven, ink):
    expected = preprocess(draw_seven())
    result = preprocess(draw_seven(ink=ink, background=(0, 0, 0, 0), mode="RGBA"))
    np.testing.assert_allclose(result, expected, atol=0.02)


def test_large_images_are_handled(draw_seven):
    result = preprocess(draw_seven(canvas=2000, left=400, top=300, size=1200))
    assert result.shape == (28, 28)
    cy, cx = centre_of_mass(result)
    assert abs(cy - 13.5) <= 1 and abs(cx - 13.5) <= 1
