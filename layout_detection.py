#!/usr/bin/env python3
"""
Layout Detection Module
Detects table layouts in document images using OpenCV
"""

import cv2
import numpy as np
import json
import os
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LayoutDetector:
    """Table layout detection using OpenCV"""
    
    def __init__(self, input_dir: str, output_dir: str):
        self.input_dir = input_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def detect_table_cells(self, image_path: str) -> List[Dict]:
        """
        Detect table cells in an image
        
        Args:
            image_path: Path to the image file
            
        Returns:
            List of detected cells with coordinates and type
        """
        try:
            filename = os.path.basename(image_path)
            logger.info(f"Analyzing table layout: {filename}")
            
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Cannot read image: {filename}")
                return []
            
            H, W, _ = image.shape
            debug_img = image.copy()
            
            # Preprocessing
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            thresh = cv2.adaptiveThreshold(~gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                         cv2.THRESH_BINARY, 15, -2)
            
            # Detect horizontal and vertical lines
            h_ker = cv2.getStructuringElement(cv2.MORPH_RECT, (W // 40, 1))
            v_ker = cv2.getStructuringElement(cv2.MORPH_RECT, (1, H // 40))
            
            mask_h = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, h_ker)
            mask_v = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, v_ker)
            mask = cv2.add(mask_h, mask_v)
            
            # Extract cells
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter cells (minimum 30x15 px)
            cells = []
            for c in contours:
                x, y, w, h = cv2.boundingRect(c)
                if w > 30 and h > 15:
                    cells.append((x, y, x + w, y + h))
            
            # Sort cells: Top to Bottom, then Left to Right
            cells.sort(key=lambda b: (b[1], b[0]))
            
            # Identify structure (Headers vs Cells)
            table_data = []
            if cells:
                # Define header as first row (20px tolerance)
                first_y = cells[0][1]
                
                for (x1, y1, x2, y2) in cells:
                    is_header = abs(y1 - first_y) < 20
                    cell_type = "header" if is_header else "cell"
                    
                    table_data.append({
                        "type": cell_type,
                        "box": [int(x1), int(y1), int(x2), int(y2)]
                    })
                    
                    # Visualization (Yellow for Header, Green for Cell)
                    color = (0, 255, 255) if is_header else (0, 255, 0)
                    cv2.rectangle(debug_img, (x1, y1), (x2, y2), color, 3)
            
            # Save debug image
            debug_path = os.path.join(self.output_dir, f"debug_{filename}")
            cv2.imwrite(debug_path, debug_img)
            
            logger.info(f"Detected {len(cells)} cells")
            return table_data
            
        except Exception as e:
            logger.error(f"Error detecting layout in {image_path}: {e}")
            return []
    
    def process_image(self, image_path: str) -> bool:
        """
        Process a single image and save layout JSON
        
        Args:
            image_path: Path to the image file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            filename = os.path.basename(image_path)
            
            # Load image to get dimensions
            image = cv2.imread(image_path)
            if image is None:
                return False
            
            H, W, _ = image.shape
            
            # Detect table cells
            table_data = self.detect_table_cells(image_path)
            
            # Create result JSON
            json_filename = os.path.splitext(filename)[0] + ".json"
            json_path = os.path.join(self.output_dir, json_filename)
            
            result_json = {
                "image": filename,
                "dimensions": {"width": W, "height": H},
                "table_found": len(table_data) > 0,
                "cells": table_data
            }
            
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(result_json, f, indent=4, ensure_ascii=False)
            
            logger.info(f"Layout JSON saved: {json_filename}")
            return True
            
        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
            return False
    
    def process_all_images(self) -> int:
        """
        Process all images in the input directory
        
        Returns:
            int: Number of successfully processed images
        """
        image_extensions = (".png", ".jpg", ".jpeg")
        image_files = [f for f in os.listdir(self.input_dir) 
                      if f.lower().endswith(image_extensions)]
        
        if not image_files:
            logger.warning(f"No images found in {self.input_dir}")
            return 0
        
        logger.info(f"Processing {len(image_files)} images...")
        successful = 0
        
        for filename in sorted(image_files):
            image_path = os.path.join(self.input_dir, filename)
            if self.process_image(image_path):
                successful += 1
        
        logger.info(f"Successfully processed {successful}/{len(image_files)} images")
        return successful


def main():
    """Main function"""
    input_dir = os.getenv('IMAGE_DIR', './images_pages')
    output_dir = os.getenv('LAYOUT_DIR', './layout_json_results')
    
    logger.info("Starting layout detection...")
    
    detector = LayoutDetector(input_dir, output_dir)
    successful_count = detector.process_all_images()
    
    if successful_count > 0:
        logger.info(f"Layout detection completed! {successful_count} files processed.")
        logger.info(f"JSON files saved to: {output_dir}")
    else:
        logger.error("Layout detection failed!")
        exit(1)


if __name__ == "__main__":
    main()
