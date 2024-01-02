import cv2
import os
import fitz  # PyMuPDF
import numpy as np
import sys
from PIL import Image

# Function to process a PDF file and save the processed images to a specified directory

def decode_predictions(scores, geometry):
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []

    for y in range(0, numRows):
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        for x in range(0, numCols):
            if scoresData[x] < 0.5:
                continue

            (offsetX, offsetY) = (x * 4.0, y * 4.0)
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])

    return (rects, confidences)

def non_max_suppression(boxes, probs=None, overlapThresh=0.3):
    if len(boxes) == 0:
        return []

    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    pick = []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    if probs is not None:
        idxs = np.argsort(probs)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[int(i)], x1[idxs[:last]])
        yy1 = np.maximum(y1[int(i)], y1[idxs[:last]])
        xx2 = np.minimum(x2[int(i)], x2[idxs[:last]])
        yy2 = np.minimum(y2[int(i)], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / area[idxs[:last]]

        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlapThresh)[0])))

    return boxes[pick].astype("int")

def process_pdf(pdf_path, output_directory):
    net = cv2.dnn.readNet(r"C:\Program Files\frozen_east_text_detection.pb")

    pdf_document = fitz.open(pdf_path)
    for page_number in range(pdf_document.page_count):
        page = pdf_document[page_number]
        pixmap = page.get_pixmap()
        if pixmap is not None:
            if pixmap.samples is not None:
                pil_image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
                image = np.array(pil_image)
                orig = image.copy()
                (H, W) = image.shape[:2]

                # Ensure the new width and height are multiples of 32
                newW = int(W / 32) * 32
                newH = int(H / 32) * 32

                rW = W / float(newW)
                rH = H / float(newH)

                blob = cv2.dnn.blobFromImage(image, 1.0, (newW, newH),
                    (123.68, 116.78, 103.94), swapRB=True, crop=False)
                
                net.setInput(blob)
                (scores, geometry) = net.forward(["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"])

                (rects, confidences) = decode_predictions(scores, geometry)
                boxes = non_max_suppression(np.array(rects), probs=confidences)

                for (startX, startY, endX, endY) in boxes:
                    startX = int(startX * rW)
                    startY = int(startY * rH)
                    endX = int(endX * rW)
                    endY = int(endY * rH)

                    cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)

                output_path = os.path.join(output_directory, f"page{page_number + 1}_processed.png")
                cv2.imwrite(output_path, orig)



if __name__ == "__main__":
    # Define the path to the PDF file
    # pdf_file_path = r"C:\Users\Adeeba\Downloads\2023060775.pdf"
    # pdf_file_path = r"C:\Users\Adeeba\Desktop\py script\Scholarship_notice_2023-24_enclosures.pdf"
    pdf_file_path = r"C:\Users\Adeeba\Downloads\testpdf_merged.pdf"

    # Define the path to the output directory
    output_directory = r"C:\Users\Adeeba\Desktop\err3dataset"

    # Process the PDF file
    process_pdf(pdf_file_path, output_directory)
    
    print("PDF PROCESSED SUCCESSFULLY")


