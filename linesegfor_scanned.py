import cv2
import os
import fitz  # PyMuPDF
import numpy as np
import sys
from PIL import Image

# Line Splitting Function....
def line_splitter(input_image):
    # Convert the input image to grayscale
    input_image_gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

    # Main list to store the output ligatures i.e. a list of numpy arrays.
    out_image_array = []

    # Variable for storing the total number of lines.
    total_lines = 0

    # Extracting the dimensions of the image
    height, width = input_image_gray.shape

    # Making a bool type filter
    filter = np.ones(shape=(1, width), dtype=bool)

    # Converting the source image to bool.
    input_image_bool = input_image_gray.astype(bool)

    # Some workplace variables for the following for loop.
    row = 0  # This variable index the pixels of the image row-wise.
    first_row = True
    line_detected = False

    # This for loop iterates through each pixel-wide line of the page and separates them.
    for i in input_image_bool:
        i = np.reshape(i, newshape=(1, width))
        res = np.bitwise_and(i, filter)
        res = np.bitwise_and.reduce(res, axis=1)

        if res == False:
            line_detected = True
            if first_row == True:
                row_start = row
                first_row = False
            row = row + 1
            continue

        if line_detected == True:
            row_end = row
            line_detected = False
            first_row = True
            out_image = input_image_bool[row_start:row_end, :]  # Cropping the image.

            # Check if the processed image is not empty
            if np.any(out_image):
                out_image = out_image.astype(int)  # Converting back to int.
                out_image = out_image * 255  # Replacing the 1's by 255's

                # Check if the processed image shape is not inhomogeneous
                if out_image.shape[1] == width:
                    # Writing the output.
                    out_image_array.append(out_image)
                    total_lines = total_lines + 1

        row = row + 1

    # Returning the list as is, instead of converting to a numpy array.
    return out_image_array

#Function for performing the task of steretching..
def stretcher(input_image,degree_of_stretch):
    # Getting the dimensions
    height_original, width_original = input_image.shape

    # Making the height of the image divisible by degree_of_stretch
    while(height_original%degree_of_stretch!=0):
        height_original=height_original+1
        input_image=np.insert(input_image,0,255,0)


    # degree_of_stretch:
    # How many pixels to skip for the shifting operation.
    # OR we can say that it is the degree of stretch.
    final_width = width_original + int(height_original / degree_of_stretch)
    img_modified = np.ones((height_original, final_width))
    img_modified=img_modified*255
    counter = 0
    inserts = 0    #paddings
    for row in img_modified:
        if counter % degree_of_stretch == 0:
            inserts = inserts + 1
        row[0 + inserts: width_original + inserts] = input_image[counter] #  row[2:22] replace by input_image[1]
        counter = counter + 1
    return img_modified


# Function for reversing the task of steretching..
def deStretcher(input_image,degree_of_stretch):
    # Getting the dimensions
    height_original, width_original = input_image.shape

    # degree_of_stretch:
    # How to many pixels to skip for the shifting operation.
    # OR we can say that it is the degree of stretch.

    # # Making the height of the image divisible by degree_of_stretch
    # while (height_original % degree_of_stretch != 0):
    #     height_original = height_original + 1
    #     original_image = np.insert(original_image, 0, 255, 0)

    final_width = width_original + int(height_original/ degree_of_stretch)
    img_modified = np.ones((height_original, final_width))
    img_modified = img_modified * 255
    counter = 0
    inserts = 0
    for row in img_modified:
        if counter % degree_of_stretch == 0:
            inserts = inserts + 1
        row[final_width - width_original - inserts:final_width - inserts] = input_image[counter]
        counter = counter + 1

    # Returning the result..
    return img_modified


# The main function for carrying out the fitting task.
def fitter(input_image):
    # Extracting the dimensions of the image
    height, width = input_image.shape
    # Making a bool type filter
    filter = np.ones(shape=(1, width), dtype=bool)
    # Converting the source image to bool.
    input_image = input_image.astype(bool)
    # Some workplace variables for the following for loop.
    row = 0  # This variable indexs the pixels of the image row wisE FROM THE TOP.
    # This for loop iterates through each pixel wide line of the page and separates them.
    # First we start from the top of the image
    for i in input_image:
        i = np.reshape(i, newshape=(1, width))
        res = np.bitwise_and(i, filter)
        res = np.bitwise_and.reduce(res, axis=1)
        if res == False:
            row_start = row
            break
        row = row + 1
        continue
    # Flipping the image horizontally.
    input_image = input_image.astype(int)  # Converting back to int.
    input_image = input_image * 255  # Replacing the 1's by 255's.
    input_image = cv2.flip(src=input_image, flipCode=0)
    input_image = input_image.astype(bool)  # Converting back to bool after flipping.
    row = 0  # This variable indexs the pixels of the image row wisE FROM THE BOTTOM.
    # Then we apply the same algorithm on the inverted image.
    for i in input_image:
        i = np.reshape(i, newshape=(1, width))
        res = np.bitwise_and(i, filter)
        res = np.bitwise_and.reduce(res, axis=1)
        if res == False:
            row_end = height - row
            break
        row = row + 1
        continue
    # Flipping the image back.
    input_image = input_image.astype(int)  # Converting back to int.
    input_image = input_image * 255  # Replacing the 1's by 255's.
    input_image = cv2.flip(src=input_image, flipCode=0)
    # Cutting the image coordinates.
    out_image = input_image[row_start:row_end, :]
    # Returning the numpy array.
    return out_image


# Function to process a PDF file and save the processed images to a specified directory
import fitz
import cv2
import numpy as np
from PIL import Image
import os

# def deskew(image):
#     coords = np.column_stack(np.where(image > 0))
#     angle = cv2.minAreaRect(coords)[-1]
#     if angle < -45:
#         angle = -(90 + angle)
#     else:
#         angle = -angle
#     (h, w) = image.shape[:2]
#     center = (w // 2, h // 2)
#     M = cv2.getRotationMatrix2D(center, angle, 1.0)
#     rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
#     return rotated
def process_pdf(pdf_path, output_directory):
    # Open the PDF file
    pdf_document = fitz.open(pdf_path)
    
    # Loop through each page in the PDF
    for page_number in range(pdf_document.page_count):
        page = pdf_document[page_number]
        pixmap = page.get_pixmap()
        
        # Check if the page is not blank
        if pixmap is not None:
            if pixmap.samples is not None:
                # Convert the page to an image
                pil_image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
                
                # Convert the image to a format that OpenCV can work with
                image = np.array(pil_image)
                
                # Convert the image to grayscale (black and white)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                
                # Reduce noise in the image
                gray = cv2.medianBlur(gray, 5)
                
                # Convert the grayscale image to binary (black and white only)
                _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                
                # Define a rectangular kernel for morphological operations
                page_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18))
                
                # Dilate (expand) the areas of the image that are white
                page_dilation = cv2.dilate(thresh, page_kernel, iterations=1)
                
                # Find the contours (outlines) of the areas of the image that are white
                page_contours, _ = cv2.findContours(page_dilation, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                
                # Initialize the minimum and maximum x and y values for the bounding box
                x_min, y_min, x_max, y_max = float('inf'), float('inf'), 0, 0
                
                # Loop through each contour
                for page_cnt in page_contours:
                    # Get the x, y, width, and height of the bounding box of the contour
                    x, y, w, h = cv2.boundingRect(page_cnt)
                    
                    # Ignore contours that are too small
                    if h < 50 or w < 50:
                        continue
                    
                    # Update the minimum and maximum x and y values
                    x_min = min(x_min, x)
                    y_min = min(y_min, y)
                    x_max = max(x_max, x + w)
                    y_max = max(y_max, y + h)
                    
                    # Draw a yellow rectangle around the contour
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 255), 1)
                    
                    # Define a rectangular kernel for morphological operations
                    para_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w, 15))
                    
                    # Dilate (expand) the areas of the image that are white
                    para_dilation = cv2.dilate(thresh[y:y+h, x:x+w], para_kernel, iterations=1)
                    
                    # Find the contours (outlines) of the areas of the image that are white
                    para_contours, _ = cv2.findContours(para_dilation, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                    
                    # Loop through each contour
                    for para_cnt in para_contours:
                        # Get the x, y, width, and height of the bounding box of the contour
                        px, py, pw, ph = cv2.boundingRect(para_cnt)
                        
                        #Ignore contours that are too small
                        if ph < 20:  
                            continue
                        
                        # Draw a green rectangle around the contour
                        cv2.rectangle(image, (x + px, y + py), (x + px + pw, y + py + ph), (0, 255, 0), 1)
                        
                        # Define a rectangular kernel for morphological operations
                        line_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (pw, 1))
                        
                        # Dilate (expand) the areas of the image that are white
                        line_dilation = cv2.dilate(thresh[y+py:y+py+ph, x+px:x+px+pw], line_kernel, iterations=1)
                        
                        # Find the contours (outlines) of the areas of the image that are white
                        line_contours, _ = cv2.findContours(line_dilation, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                        # cv2.imwrite(r'C:\Users\Adeeba\Desktop\datasetgen_script\updatedurdu.py', line_contours)
                        # Loop through each contour
                        for line_cnt in line_contours:
                            # Get the x, y, width, and height of the bounding box of the contour
                            lx, ly, lw, lh = cv2.boundingRect(line_cnt)
                            
                            # Ignore contours that are too small
                            if lh < 10: 
                                continue
                            
                            # Draw a red rectangle around the contour
                            cv2.rectangle(image, (x + px + lx, y + py + ly), (x + px + lx + lw, y + py + ly + lh), (0, 0, 255), 1)
                
                # Draw a yellow rectangle around the entire page
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)
                
                # Save the image
                output_path = os.path.join(output_directory, f"page{page_number + 1}_processed.png")
                cv2.imwrite(output_path, image)

if __name__ == "__main__":
    # Define the path to the PDF file
    # pdf_file_path = r"C:\Users\Adeeba\Downloads\2023060775.pdf"
    # pdf_file_path = r"C:\Users\Adeeba\Desktop\py script\Scholarship_notice_2023-24_enclosures.pdf"
    pdf_file_path = r"PDF Path"
    output_directory = r"path-to-preferred-output-directory"
    # Process the PDF file
    process_pdf(pdf_file_path, output_directory)
    
    print("PDF PROCESSED SUCCESSFULLY")
