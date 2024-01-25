import cv2 
import numpy as np
import os
import math

def get_string(img_path):
    
    img = cv2.imread(img_path)

    img = cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)

    rgb_planes = cv2.split(img)
    result_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
        # bg_img = cv2.medianBlur(dilated_img, 15)
        bg_img = cv2.GaussianBlur(dilated_img, (5,5),0)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        result_planes.append(diff_img)
    img = cv2.merge(result_planes)
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # kernel = np.ones((2, 2), np.uint8)
    # img = cv2.dilate(img, kernel, iterations=1)
    # img = cv2.erode(img, kernel, iterations=1) 

    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    # kernel_line = np.ones((5, 5), np.uint8)
    # img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel_line)
    processed_img_path = 'processed.png'
    cv2.imwrite(processed_img_path, img)
    
    draw_contours(processed_img_path)

def draw_contours(img_path): 
    img = cv2.imread(img_path,0)
    original_img = img.copy() 
    
    img = cv2.GaussianBlur(img, (5, 5), 0)

    # Threshold the image
    # _, threshed = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    threshed = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, 11, 1)

    # Remove vertical table borders
    lines = cv2.HoughLinesP(threshed, 1, np.pi/90, 40, minLineLength=40, maxLineGap=10)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Calculate the angle of the line
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            # Ignore lines that are vertical or near-vertical
            if abs(angle) > 45:
                cv2.line(threshed, (x1, y1), (x2, y2), (0, 0, 0), 6)        
    lines_removed=threshed 

    # Remove horizontal table borders
    kernel = np.ones((5, 1), np.uintp) 
    opened = cv2.morphologyEx(lines_removed, cv2.MORPH_OPEN, kernel)

    #perform erosion for noise removal
    kernel = np.ones((2,2), np.uint8)
    erosion = cv2.erode(opened, kernel)

    #dialation
    kernel = np.ones((3,50), np.uint8)
    dilated = cv2.dilate(erosion, kernel, iterations=1)

    #closing to fill the smalls gaps left by dialation
    kernel = np.ones((5,15), np.uint8)
    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_contours = original_img.copy()
    cv2.drawContours(img_contours, contours, -1, (0, 255, 0), 1)
    cv2.imwrite(os.path.join(output_dir, 'contours.png'), img_contours)

    x1=1
    boxes = []
    for i, cnt in enumerate(contours):
        if cv2.contourArea(cnt) > 800:  # Set a minimum contour area
            _,_,w,h = cv2.boundingRect(cnt)
            aspect_ratio = float(w)/h
            if (1.5<aspect_ratio): 
                if h < img.shape[0] * 0.7 and w < img.shape[0] * 0.8:   # Avoid very large boxes that span most of the image height and width
                    rect = cv2.minAreaRect(cnt)
                    box = cv2.boxPoints(rect)
                    box = np.intp(box)
                    
                    # Increase top and bottom height
                    top_increase_factor = 4   
                    bottom_increase_factor = 3  
                    center_y = sum([point[1] for point in box]) / 4
                    for point in box:
                        if point[1] < center_y:
                            point[1] -= top_increase_factor
                        else:
                            point[1] += bottom_increase_factor

                    width = int(rect[1][0])
                    height = int(rect[1][1]) + (top_increase_factor + bottom_increase_factor) 

                    src_pts = box.astype("float32")
                    # Coordinate of the points in box points after the rectangle has been straightened
                    dst_pts = np.array([[0, height-1],
                                        [0, 0],
                                        [width-1, 0],
                                        [width-1, height-1]], dtype="float32")

                    # The perspective transformation matrix
                    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
                    warped = cv2.warpPerspective(original_img, M, (width, height))

                    if height > width:
                        warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)

                    # Save the cropped image
                    cv2.imwrite(os.path.join(output_dir, f'line_{x1}.png'), warped)
                    x1+=1
                    boxes.append(box) 

    # image with bounding boxes
    for box in boxes:
        cv2.drawContours(original_img,[box],0,(0,255,0),1, lineType=cv2.LINE_AA)
    filename = os.path.join(output_dir, 'bounding_boxes.png')
    cv2.imwrite(filename, original_img)
    print('Saved as', filename)
    return img

img_path = r"image-path"
output_dir =r"path-to-preferred-output-directory"
get_string(img_path)
