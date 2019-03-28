from PIL import ImageDraw
from PIL import ImageFont


def show_bboxes(img, bounding_boxes=None, facial_landmarks=[]):
    """Draw bounding boxes and facial landmarks.

    Arguments:
        img: an instance of PIL.Image.
        bounding_boxes: a float numpy array of shape [n, 5].
        facial_landmarks: a float numpy array of shape [n, 10].

    Returns:
        an instance of PIL.Image.
    """

    img_copy = img.copy()
    draw = ImageDraw.Draw(img_copy)
#     for b in bounding_boxes:
#         draw.rectangle([
#             (b[0], b[1]), (b[2], b[3])
#         ], outline='white')

    for p in facial_landmarks:
            for i in range(106):
                draw.ellipse([
                    (p[i*2] - 1.0, p[2*i + 1] - 1.0),
                    (p[i*2] + 1.0, p[2*i+1] + 1.0)
                ], outline='blue')
                font = ImageFont.truetype("arial.ttf", 10)
                draw.text([p[2*i], p[2*i+1]], str(i), font=font)

    return img_copy
