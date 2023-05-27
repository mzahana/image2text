from easyocr import Reader
import argparse
import cv2

parser = argparse.ArgumentParser()
parser.add_argument("--image", required=True,
                    help="path to input image")
parser.add_argument("--langs", type=str, default="en",
                    help="comma separated list of languages for our OCR")
parser.add_argument("-g", "--gpu", type=int, default=-1,
                    help="whether or not GPU should be used")
args = vars(parser.parse_args())

langs = args["langs"].split(",")
print("[INFO] Using the following languages: {}".format(langs))
# load the input image from disk
image = cv2.imread(args["image"])
# OCR the input image using EasyOCR
print("[INFO] Performing OCR on the input image")
reader = Reader(langs, gpu=args["gpu"] > 0)
results = reader.readtext(image)

for (bbx, text, prob) in results:
    print("[INFO] {:.4f}: {}".format(prob, text))
    # unpack the bounding box
    (top_left, top_right, bottom_right, bottom_left) = bbx
    tl = (int(top_left[0]), int(top_left[1]))
    tr = (int(top_right[0]), int(top_right[1]))
    br = (int(bottom_right[0]), int(bottom_right[1]))
    bl = (int(bottom_left[0]), int(bottom_left[1]))

    cv2.rectangle(image, tl, br, (0, 0, 255), 2)
    cv2.putText(image, text, (tl[0], tl[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
cv2.imshow("image", image)
cv2.waitKey(0)