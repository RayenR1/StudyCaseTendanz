#!/usr/bin/env python3
"""
PDF to Images Converter
Converts PDF pages to PNG images using PyMuPDF
"""

import os
import fitz  
from PIL import Image
import io
from pathlib import Path
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def pdf_to_images(pdf_path: str, output_folder: str) -> bool:
    """
    Convert PDF pages to PNG images
    
    Args:
        pdf_path: Path to the PDF file
        output_folder: Directory to save images
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        if not os.path.exists(pdf_path):
            logger.error(f"PDF file not found: {pdf_path}")
            return False
        
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            logger.info(f"Created output directory: {output_folder}")
        
        # Open PDF with PyMuPDF
        pdf_document = fitz.open(pdf_path)
        logger.info(f"Opened PDF: {pdf_path} with {len(pdf_document)} pages")
        
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            
            # Convert to image with 300 DPI
            mat = fitz.Matrix(300/72, 300/72)
            pix = page.get_pixmap(matrix=mat)
            
            # Convert to PIL Image
            img_data = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
            
            # Save image
            page_number = page_num + 1
            image_name = f"{page_number}.png"
            save_path = os.path.join(output_folder, image_name)
            img.save(save_path, 'PNG')
            
            logger.info(f"Page {page_number} saved as {image_name}")
        
        pdf_document.close()
        logger.info(f"All pages converted and saved to '{output_folder}' folder")
        return True
        
    except Exception as e:
        logger.error(f"Error converting PDF to images: {e}")
        return False


def main():
    """Main function"""
    pdf_path = os.getenv('PDF_PATH', './Echantillon_2.pdf')
    output_folder = os.getenv('IMAGE_DIR', './images_pages')
    
    logger.info("Starting PDF to Images conversion...")
    success = pdf_to_images(pdf_path, output_folder)
    
    if success:
        logger.info("PDF conversion completed successfully!")
    else:
        logger.error("PDF conversion failed!")
        exit(1)


if __name__ == "__main__":
    main()
