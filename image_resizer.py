#!/usr/bin/env python3
"""
Image Resizer Module
Resizes images to optimize them for Gemini API processing
"""

import os
import logging
from PIL import Image
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ImageResizer:
    """Image resizer for optimizing images before processing"""
    
    def __init__(self, image_dir: str, backup_dir: Optional[str] = None, 
                 max_pixels: int = 80_000_000, min_width: int = 1200):
        self.image_dir = image_dir
        self.backup_dir = backup_dir
        self.max_pixels = max_pixels
        self.min_width = min_width
        
        if backup_dir and not os.path.exists(backup_dir):
            os.makedirs(backup_dir)
    
    def resize_image_if_needed(self, img_path: str) -> bool:
        """
        Resize image if it exceeds maximum pixels
        
        Args:
            img_path: Path to the image file
            
        Returns:
            bool: True if image was resized, False if no resize needed
        """
        try:
            with Image.open(img_path) as img:
                original_width, original_height = img.size
                original_pixels = original_width * original_height
                
                filename = Path(img_path).name
                logger.info(f"{filename} → {original_width}x{original_height} "
                          f"({original_pixels // 1_000_000}M pixels)", extra={'end': ' '})
                
                if original_pixels <= self.max_pixels:
                    logger.info("→ OK (no resize needed)")
                    return False
                
                # Calculate reduction ratio
                ratio = (self.max_pixels / original_pixels) ** 0.5
                new_width = max(int(original_width * ratio), self.min_width)
                new_height = int(new_width * original_height / original_width)
                
                logger.info(f"→ RESIZING → {new_width}x{new_height} "
                          f"({(new_width * new_height) // 1_000_000}M pixels)")
                
                # Resize with best quality
                img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                
                # Backup original if specified
                if self.backup_dir:
                    backup_path = os.path.join(self.backup_dir, filename)
                    img.save(backup_path)
                    logger.info(f"Original backed up to: {backup_path}")
                
                # Save resized image
                img_resized.save(img_path, "PNG", optimize=True)
                return True
                
        except Exception as e:
            logger.error(f"Error resizing {img_path}: {e}")
            return False
    
    def resize_all_images(self) -> tuple[int, int]:
        """
        Resize all images in the directory
        
        Returns:
            tuple: (total_images, resized_count)
        """
        if not os.path.exists(self.image_dir):
            logger.error(f"Image directory does not exist: {self.image_dir}")
            return 0, 0
        
        image_files = [f for f in os.listdir(self.image_dir)
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        
        if not image_files:
            logger.warning("No images found in directory")
            return 0, 0
        
        logger.info(f"Analyzing {len(image_files)} images in '{self.image_dir}'...\n")
        
        resized_count = 0
        for img_name in sorted(image_files):
            img_path = os.path.join(self.image_dir, img_name)
            if self.resize_image_if_needed(img_path):
                resized_count += 1
        
        return len(image_files), resized_count


def main():
    """Main function"""
    image_dir = os.getenv('IMAGE_DIR', './images_pages')
    max_pixels = int(os.getenv('MAX_PIXELS', '80000000'))
    min_width = int(os.getenv('MIN_WIDTH', '1200'))
    
    logger.info("Starting image optimization...")
    
    resizer = ImageResizer(
        image_dir=image_dir,
        backup_dir=None,  # No backup by default
        max_pixels=max_pixels,
        min_width=min_width
    )
    
    total_images, resized_count = resizer.resize_all_images()
    
    print("\n" + "="*60)
    if resized_count == 0:
        logger.info("No images needed resizing.")
        logger.info("You can run the Gemini processing directly!")
    else:
        logger.info(f"Result: {resized_count}/{total_images} images resized.")
        logger.info("All images are now optimized and readable.")
        logger.info("You can safely run the Gemini extraction pipeline!")
    print("="*60)


if __name__ == "__main__":
    main()
