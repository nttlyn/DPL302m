from ultralytics import YOLO
import cv2
import math
import time
import os
import readplate
def is_bbox_inside(b1, b2):
    return b2[0] <= b1[0] and b2[1] <= b1[1] and b2[2] >= b1[2] and b2[3] >= b1[3]

def video_detection(path_x):
    video_capture = path_x
    cap = cv2.VideoCapture(video_capture)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 25, (frame_width, frame_height))

    model = YOLO("best_2.pt")
    classNames = ["vehicle", "helmet", "no-helmet", "plate"]

    frame_count = 0
    frame_skip = 12
    evidence_folder = 'evidence'
    if not os.path.exists(evidence_folder):
        os.makedirs(evidence_folder)
    plate_folder = 'plate'
    if not os.path.exists(plate_folder):
        os.makedirs(plate_folder)
#    captured_plates = set()

    while True:
        ret, img = cap.read()

        if not ret:
            print("Error: Failed to read frame.")
            break

        frame_count += 1

        if frame_count % frame_skip != 0:
            continue

        results = model(img, stream=True, verbose=False)

        vehicles = []
        plates = []
        non_helmets = []

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                class_name = classNames[cls]

                if class_name == "vehicle":
                    vehicles.append((x1, y1, x2, y2))
                elif class_name == "plate":
                    plates.append((x1, y1, x2, y2))
                elif class_name == "no-helmet":
                    non_helmets.append((x1, y1, x2, y2))

        for vehicle in vehicles:
            for plate in plates:
                for non_helmet in non_helmets:
                    if is_bbox_inside(plate, vehicle) and is_bbox_inside(non_helmet, vehicle):
                        #if plate not in captured_plates:
                            vehicle_img=img[vehicle[1]:vehicle[3],vehicle[0]:vehicle[2]]
                            plate_img = img[plate[1]:plate[3], plate[0]:plate[2]]
                            name_result=readplate.show(plate_img)

                            folder_evidence = os.path.join(evidence_folder,f'vehicle_{name_result}.jpg')
                            cv2.imwrite(folder_evidence, vehicle_img)

                            folder_plate = os.path.join(plate_folder, f'plate_{name_result}.jpg')
                            cv2.imwrite(folder_plate, plate_img)
                            
                            #captured_plates.add(plate)


                            cv2.rectangle(img, (vehicle[0], vehicle[1]), (vehicle[2], vehicle[3]), (255, 0, 255), 1)
                            cv2.rectangle(img, (plate[0], plate[1]), (plate[2], plate[3]), (255, 0, 255), 1)
                            cv2.rectangle(img, (non_helmet[0], non_helmet[1]), (non_helmet[2], non_helmet[3]), (255, 0, 255), 1)
                            
                            label_plate = f'plate{conf}'
                            label_no_helmet = f'no-helmet{conf}'
                            label_vehicle = f'vehicle{conf}'

                            cv2.putText(img, label_plate, (plate[0], plate[1]-2), 0, 1, [255,255,255], thickness=1, lineType=cv2.LINE_AA)
                            cv2.putText(img, label_no_helmet, (non_helmet[0], non_helmet[1]-2), 0, 1, [255,255,255], thickness=1, lineType=cv2.LINE_AA)
                            cv2.putText(img, label_vehicle, (vehicle[0], vehicle[1]-2), 0, 1, [255,255,255], thickness=1, lineType=cv2.LINE_AA)


        yield img
        out.write(img)
        cv2.imshow("image", img)
        out.write(img)

        if cv2.waitKey(1) & 0xFF == ord('1'):
            break

    out.release()
cv2.destroyAllWindows()
