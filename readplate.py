import numpy as np
import cv2
from tensorflow.keras.models import Sequential, load_model

def segment_characters(img) :
    img_lp = cv2.resize(img, (275, 184))
    img_gray_lp = cv2.cvtColor(img_lp, cv2.COLOR_BGR2GRAY)
    #img_gray_lp = cv2.equalizeHist(img_gray_lp)
    _, img_binary_lp = cv2.threshold(img_gray_lp, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    img_binary_lp = cv2.erode(img_binary_lp, (3,3))
    img_binary_lp = cv2.dilate(img_binary_lp, (3,3))
    
   # LP_HEIGHT = img_binary_lp.shape[0]
   # LP_WIDTH = img_binary_lp.shape[1]

    img_binary_lp[:5 :] = 255
    img_binary_lp[-5:, :] = 255
    img_binary_lp[:, :5] = 255
    img_binary_lp[:, -5:] = 255

    dimensions = [20,
                       92,
                       27.5,
                       183.3]
    cv2.imwrite('contour.jpg',img_binary_lp)

    char_list = find_contours(dimensions, img_binary_lp)

    return char_list
def find_contours(dimensions, img) :

    cntrs, _ = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    lower_width = dimensions[0]
    upper_width = dimensions[1]
    lower_height = dimensions[2]
    upper_height = dimensions[3]
    
    cntrs = sorted(cntrs, key=cv2.contourArea, reverse=True)[:13]
    
    ii = cv2.imread('contour.jpg')
    
    x_cntr_list = []
    target_contours = []
    img_res = []
    for cntr in cntrs :
        intX, intY, intWidth, intHeight = cv2.boundingRect(cntr)
        
        if intWidth > lower_width and intWidth < upper_width and intHeight > lower_height and intHeight < upper_height :
            x_cntr_list.append(intX)
            
            char_copy = np.zeros((44,24))
            char = img[intY:intY+intHeight, intX:intX+intWidth]
            char = cv2.resize(char, (20, 40))
            
            cv2.rectangle(ii, (intX,intY), (intWidth+intX, intY+intHeight), (50,21,200), 2)
            center_y = intY + intHeight // 2
            char = cv2.subtract(255, char)
            char_copy[2:42, 2:22] = char
            char_copy[0:2, :] = 0
            char_copy[:, 0:2] = 0
            char_copy[42:44, :] = 0
            char_copy[:, 22:24] = 0

            img_res.append((center_y, intX, char_copy)) 
          
    img_res = sorted(img_res, key=lambda x: (x[0] > 92, x[0] < 92, x[1]))
    img_res = [char[2] for char in img_res]
    img_res = np.array(img_res)

    return img_res   
model=load_model('best_model_3.h5')
def fix_dimension(image): 
    img = np.zeros((28, 28, 3))
    for i in range(3):
        img[:,:,i] = image
    return img
def show_results(char):
    dic = {}
    characters = '0123456789ABCDEFGHKLMNPSTUVXYZ'
    for i,c in enumerate(characters):
        dic[i] = c

    output = []
    for i, ch in enumerate(char):
        image_ = cv2.resize(ch, (28, 28), interpolation=cv2.INTER_AREA)
        image = fix_dimension(image_)
        image = image.reshape(1, 28, 28, 3)
        predictions = model.predict(image)
        y_ = np.argmax(predictions[0])
        characters = dic[y_]
        output.append(characters)
        
    plate_number = ''.join(output)
    return plate_number
def show(img):
    char = segment_characters(img)
    plate_number = show_results(char)
    return plate_number
