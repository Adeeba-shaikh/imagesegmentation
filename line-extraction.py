import cv2
import numpy as np

def extract_lines(image_path, output_folder):
    # Read the image
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding 
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = np.ones((1, 40), np.uint8)
    img_dilated = cv2.dilate(binary, kernel, iterations=1)
    # blur = cv2.GaussianBlur(img_dilated, (5,5), cv2.BORDER_DEFAULT)
    # Find contours 
    contours, _ = cv2.findContours(img_dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(len(contours))
    line_images = []
    for contour in contours:
        min_contour_area = 300  # Adjust as needed
        min_aspect_ratio = 5    # Adjust as needed

        area = cv2.contourArea(contour)
        if area > min_contour_area:
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.intp(box)
            line = image[min(box[:, 1]):max(box[:, 1]), min(box[:, 0]):max(box[:, 0])]

            line_images.append(line)
            
    
    for i, line_img in sorted(enumerate(line_images)):
        cv2.imwrite(f'{output_folder}/line_{i + 1}.png', line_img)



if __name__ == "__main__":
    # if using CLA use this
    #if len(sys.argv) != 3:
    #     sys.exit("Usage: python lines.py filename /output-dir")
    # input_image_path = sys.argv[1]
    # output_folder_path = sys.argv[2]
    input_image_path = r'C:\NIC AI Training\PrePro\output_path\ocr\mar-test-1_filter_as.png'
    output_folder_path = r"C:\NIC AI Training\PrePro\output_path\out"
    extract_lines(input_image_path, output_folder_path)

    
