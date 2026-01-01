#!/usr/bin/env python3
"""
Main Pipeline Script
Orchestrates the complete document processing pipeline
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Import our modules
from pdf_to_images import pdf_to_images
from layout_detection import LayoutDetector
from image_resizer import ImageResizer
from gemini_extractor import GeminiExtractor

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Main document processing pipeline"""
    
    def __init__(self):
        # Load configuration from environment
        self.pdf_path = os.getenv('PDF_PATH', './Echantillon_2.pdf')
        self.image_dir = os.getenv('IMAGE_DIR', './images_pages')
        self.layout_dir = os.getenv('LAYOUT_DIR', './layout_json_results')
        self.output_dir = os.getenv('OUTPUT_DIR', './output')
        self.max_pixels = int(os.getenv('MAX_PIXELS', '80000000'))
        self.min_width = int(os.getenv('MIN_WIDTH', '1200'))
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
    
    def step_1_convert_pdf(self) -> bool:
        """Step 1: Convert PDF to images"""
        logger.info("="*60)
        logger.info("STEP 1: Converting PDF to images")
        logger.info("="*60)
        
        return pdf_to_images(self.pdf_path, self.image_dir)
    
    def step_2_detect_layout(self) -> bool:
        """Step 2: Detect table layouts"""
        logger.info("="*60)
        logger.info("STEP 2: Detecting table layouts")
        logger.info("="*60)
        
        detector = LayoutDetector(self.image_dir, self.layout_dir)
        successful_count = detector.process_all_images()
        return successful_count > 0
    
    def step_3_resize_images(self) -> bool:
        """Step 3: Resize images for optimization"""
        logger.info("="*60)
        logger.info("STEP 3: Optimizing images")
        logger.info("="*60)
        
        resizer = ImageResizer(
            image_dir=self.image_dir,
            max_pixels=self.max_pixels,
            min_width=self.min_width
        )
        total_images, resized_count = resizer.resize_all_images()
        logger.info(f"Processed {resized_count}/{total_images} images")
        return total_images > 0
    
    def step_4_extract_data(self) -> bool:
        """Step 4: Extract data using Gemini"""
        logger.info("="*60)
        logger.info("STEP 4: Extracting data with Gemini")
        logger.info("="*60)
        
        try:
            extractor = GeminiExtractor()
            return extractor.extract_from_documents()
        except Exception as e:
            logger.error(f"Gemini extraction failed: {e}")
            return False
    
    def run_pipeline(self, skip_pdf: bool = False, skip_layout: bool = False, 
                    skip_resize: bool = False, skip_extraction: bool = False) -> bool:
        """Run the complete pipeline"""
        logger.info("Starting Document Processing Pipeline")
        logger.info(f"PDF: {self.pdf_path}")
        logger.info(f"Images: {self.image_dir}")
        logger.info(f"Layouts: {self.layout_dir}")
        logger.info(f"Output: {self.output_dir}")
        
        try:
            # Step 1: PDF to Images
            if not skip_pdf:
                if not self.step_1_convert_pdf():
                    logger.error("PDF conversion failed!")
                    return False
            else:
                logger.info("Skipping PDF conversion")
            
            # Step 2: Layout Detection
            if not skip_layout:
                if not self.step_2_detect_layout():
                    logger.error("Layout detection failed!")
                    return False
            else:
                logger.info("Skipping layout detection")
            
            # Step 3: Image Resizing
            if not skip_resize:
                if not self.step_3_resize_images():
                    logger.error("Image resizing failed!")
                    return False
            else:
                logger.info("Skipping image resizing")
            
            # Step 4: Data Extraction
            if not skip_extraction:
                if not self.step_4_extract_data():
                    logger.error("Data extraction failed!")
                    return False
            else:
                logger.info("Skipping data extraction")
            
            logger.info("="*60)
            logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
            logger.info("="*60)
            return True
            
        except KeyboardInterrupt:
            logger.info("Pipeline interrupted by user")
            return False
        except Exception as e:
            logger.error(f"Pipeline failed with error: {e}")
            return False


def main():
    """Main function with command line arguments"""
    parser = argparse.ArgumentParser(description="Document Processing Pipeline")
    parser.add_argument("--skip-pdf", action="store_true", help="Skip PDF conversion")
    parser.add_argument("--skip-layout", action="store_true", help="Skip layout detection")
    parser.add_argument("--skip-resize", action="store_true", help="Skip image resizing")
    parser.add_argument("--skip-extraction", action="store_true", help="Skip data extraction")
    parser.add_argument("--only-pdf", action="store_true", help="Run only PDF conversion")
    parser.add_argument("--only-layout", action="store_true", help="Run only layout detection")
    parser.add_argument("--only-resize", action="store_true", help="Run only image resizing")
    parser.add_argument("--only-extraction", action="store_true", help="Run only data extraction")
    
    args = parser.parse_args()
    
    # Create processor
    processor = DocumentProcessor()
    
    # Handle "only" options
    if args.only_pdf:
        success = processor.step_1_convert_pdf()
    elif args.only_layout:
        success = processor.step_2_detect_layout()
    elif args.only_resize:
        success = processor.step_3_resize_images()
    elif args.only_extraction:
        success = processor.step_4_extract_data()
    else:
        # Run full pipeline with skip options
        success = processor.run_pipeline(
            skip_pdf=args.skip_pdf,
            skip_layout=args.skip_layout,
            skip_resize=args.skip_resize,
            skip_extraction=args.skip_extraction
        )
    
    if not success:
        logger.error("Pipeline execution failed!")
        sys.exit(1)
    else:
        logger.info("Pipeline execution successful!")


if __name__ == "__main__":
    main()
