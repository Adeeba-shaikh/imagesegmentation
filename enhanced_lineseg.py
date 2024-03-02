import os
import cv2
import fitz
import numpy as np
from PIL import Image
import os
import cv2
import fitz
import numpy as np
from PIL import Image

def process_pdf(pdf_path, output_directory):
    pdf_document = fitz.open(pdf_path)
    line_no = 0
    for page_number in range(pdf_document.page_count):
        page = pdf_document[page_number]
        pixmap = page.get_pixmap()
        if pixmap is not None:
            if pixmap.samples is not None:
                pil_image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
                image = np.array(pil_image)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                
                gray = cv2.medianBlur(gray, 5)
                gray = cv2.GaussianBlur(gray, (5, 5), 0)
                _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                
                page_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
                page_dilation = cv2.dilate(thresh, page_kernel, iterations=1)
                page_contours, _ = cv2.findContours(page_dilation, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                
                for page_cnt in sorted(page_contours, key=lambda cnt: cv2.boundingRect(cnt)[1]):
                    x, y, w, h = cv2.boundingRect(page_cnt)
                    
                    para_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w, 15))
                    para_dilation = cv2.dilate(thresh[y:y+h, x:x+w], para_kernel, iterations=2)
                    para_contours, _ = cv2.findContours(para_dilation, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                    
                    for para_cnt in sorted(para_contours, key=lambda cnt: cv2.boundingRect(cnt)[1]):
                        px, py, pw, ph = cv2.boundingRect(para_cnt)
                        
                        line_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (pw, 1))
                        line_dilation = cv2.dilate(thresh[y+py:y+py+ph, x+px:x+px+pw], line_kernel, iterations=1)
                        line_contours, _ = cv2.findContours(line_dilation, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                        
                        for line_cnt in sorted(line_contours, key=lambda cnt: cv2.boundingRect(cnt)[1]):
                            lx, ly, lw, lh = cv2.boundingRect(line_cnt)
                            
                            cv2.rectangle(image, (x + px + lx, y + py + ly), (x + px + lx + lw, y + py + ly + lh), (0, 0, 255), 1)
                            line_image = image[y+py+ly:y+py+ly+lh, x+px+lx:x+px+lx+lw]
                            output_path = os.path.join(output_directory, f"line{line_no}_page{page_number + 1}.png")
                            cv2.imwrite(output_path, line_image)
                            line_no += 1
                
                page_image_path = os.path.join(output_directory, f"page{page_number + 1}.png")
                cv2.imwrite(page_image_path, image)
if __name__ == "__main__":
    pdf_file_path = r"PDF Path"
    output_directory = r"path-to-preferred-output-directory"
    process_pdf(pdf_file_path, output_directory)
    print("PDF PROCESSED SUCCESSFULLY")
