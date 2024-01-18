from PIL import ImageFont, ImageDraw, Image
import numpy as np
from easyocr import Reader
import cv2
from google.colab.patches import cv2_imshow

def process_license_plate(image_path):

    img = cv2.imread(image_path)
    img = cv2.resize(img, (800, 600))
    enhanced_img = cv2.convertScaleAbs(img, alpha=1.2, beta=30)

    fontpath = "/content/Arial.ttf" 
    #link file font:        https://drive.google.com/file/d/1h3ddbT4npcCkuhU74ssBKiBezNJUNsQW/view?usp=sharing
  
    font = ImageFont.truetype(fontpath, 32)
    b, g, r, a = 0, 255, 0, 0

    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(grayscale, (5, 5), 0)
    edged = cv2.Canny(blurred, 10, 200)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    number_plate_shape = None
    for c in contours:
        perimeter = cv2.arcLength(c, True)
        approximation = cv2.approxPolyDP(c, 0.02 * perimeter, True)
        print(approximation)
        if len(approximation) == 4:  # rectangle
            number_plate_shape = approximation
            break

    if number_plate_shape is not None:
        (x, y, w, h) = cv2.boundingRect(number_plate_shape)
        number_plate = grayscale[y:y + h, x:x + w]

        reader = Reader(['en'])
        detection = reader.readtext(number_plate)

        if len(detection) == 0:
            text = "Không thấy bảng số xe"
            img_pil = Image.fromarray(img)
            draw = ImageDraw.Draw(img_pil)
            draw.text((150, 500), text, font=font, fill=(b, g, r, a))
            img = np.array(img_pil)
            cv2_imshow(img)
            cv2.waitKey(0)
        else:
            cv2.drawContours(img, [number_plate_shape], -1, (255, 0, 0), 3)
            text = "Biển số: " + f"{detection[0][1]}"
            img_pil = Image.fromarray(img)
            draw = ImageDraw.Draw(img_pil)
            draw.text((200, 500), text, font=font, fill=(b, g, r, a))
            img = np.array(img_pil)
            cv2_imshow(img)
            cv2.waitKey(0)

image_path = '/content/xetongthong.jpg' #link ảnh dô
process_license_plate(image_path)
