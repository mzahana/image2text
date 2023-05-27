import cv2
import pytesseract
from ultralytics import YOLO
import numpy as np
from ultralytics.yolo.utils.plotting import Annotator
# from plt_bbx import plot_bboxes

def plotBBX(results, img):
    for r in results:
        
        annotator = Annotator(img)
        
        boxes = r.boxes
        for box in boxes:
            
            b = box.xyxy[0]  # get box coordinates in (top, left, bottom, right) format
            c = box.cls
            annotator.box_label(b, 'label')
          
    frame = annotator.result()  
    cv2.imshow('YOLO V8 Detection', frame)     
    # if cv2.waitKey(1) & 0xFF == ord(' '):
    #     return

def detect(img: np.ndarray, model_path: str):
    model = YOLO(model_path)

    results = model.predict(source=img, save=True, save_txt=True)  # save predictions as labels
    return results

def extract_text(img: np.ndarray, boxes):
    # Convert the image to grayscale
    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    for box in boxes:
        x1 = int(box.xyxy[0].tolist()[0]) # get box coordinates in (top, left, bottom, right) format
        y1 = int(box.xyxy[0].tolist()[1])
        x2 = int(box.xyxy[0].tolist()[2])
        y2 = int(box.xyxy[0].tolist()[3])

        # Crop the image based on the bounding box coordinates
        # cropped_image = grayscale[y:y+height, x:x+width]
        cropped_image = grayscale[y1:y2, x1:x2]

        # Perform OCR using pytesseract
        text = pytesseract.image_to_string(cropped_image)

        # Overlay the extracted text on the original image
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, text, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    return img



def main(img_path: str, model_path: str):
    # Rad image file
    img = cv2.imread(img_path)

    # Detect objects
    results = detect(img, model_path)
    # plotBBX(results, img)

    # Example bounding boxes, format: (x, y, width, height)
    # bounding_boxes = [(100, 100, 200, 50), (300, 200, 150, 80)]
    result_image = extract_text(img, results[0].boxes)

    # # Display the result image
    cv2.imshow('Text Extraction', result_image)
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Press 'q' to quit
            break

if __name__ == "__main__":
    # Example usage
    image_path = '/home/user/shared_volume/Parcel-Label-Detection.v1i.yolov8/test/images/test_png_jpg.rf.efe1e48a7a1b1d576c751604709d7754.jpg'
    yolo_model_path="/home/user/shared_volume/runs/detect/train5/weights/best.pt"

    main(image_path, yolo_model_path)