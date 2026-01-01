#!/usr/bin/env python3
"""
Gemini Document Extractor
Extracts structured data from insurance documents using Google Gemini API
"""

import google.genai as genai
from PIL import Image
import json
import os
import time
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class GeminiExtractor:
    """Document extraction using Google Gemini API with multiple keys rotation"""
    
    def __init__(self):
        self.api_keys = self._load_api_keys()
        self.clients = [genai.Client(api_key=key) for key in self.api_keys]
        self.current_key_index = 0
        
        # Configuration from environment
        self.model_name = os.getenv('MODEL_NAME', 'gemini-2.5-flash')
        self.image_dir = os.getenv('IMAGE_DIR', './images_pages')
        self.layout_dir = os.getenv('LAYOUT_DIR', './layout_json_results')
        self.batch_size = int(os.getenv('BATCH_SIZE', '3'))
        self.final_file = os.getenv('FINAL_EXTRACTION_FILE', 'extraction_assurance_finale.json')
        self.backup_file = os.getenv('BACKUP_FILE', 'backup_partiel.json')
        self.max_pixels = int(os.getenv('MAX_PIXELS', '80000000'))
        
        logger.info(f"Initialized with {len(self.api_keys)} API keys")
        logger.info(f"Model: {self.model_name}, Batch size: {self.batch_size}")
    
    def _load_api_keys(self) -> List[str]:
        """Load API keys from environment variables"""
        keys = []
        for i in range(1, 7):  # Support up to 6 keys
            key = os.getenv(f'GEMINI_API_KEY_{i}')
            if key and key != 'your_api_key_here':
                keys.append(key)
        
        if not keys:
            raise ValueError("No valid Gemini API keys found in environment variables!")
        
        logger.info(f"Loaded {len(keys)} API keys")
        return keys
    
    def get_current_client(self) -> genai.Client:
        """Get current API client"""
        return self.clients[self.current_key_index]
    
    def switch_to_next_key(self):
        """Switch to next API key"""
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        logger.info(f"Switched to API key #{self.current_key_index + 1}/{len(self.api_keys)}")
    
    def resize_image_if_needed(self, img_path: str) -> Image.Image:
        """Resize image if needed for API processing"""
        img = Image.open(img_path)
        if img.width * img.height > self.max_pixels:
            ratio = (self.max_pixels / (img.width * img.height)) ** 0.5
            new_size = (int(img.width * ratio), int(img.height * ratio))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
            logger.info(f"Image resized to: {new_size}")
        return img
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for document analysis"""
        return """
Tu es un expert interne en Document Understanding travaillant pour une compagnie d'assurance française.
Tu analyses exclusivement des documents internes privés (Conditions Générales d'assurance appartenant à notre société).

CE DOCUMENT EST STRICTEMENT INTERNE ET CONFIDENTIEL — IL N'EST PAS PUBLIÉ, NI PROTÉGÉ PAR COPYRIGHT PUBLIC.
Tu dois extraire les informations de manière structurée sans jamais réciter textuellement de longs passages.
Ton objectif est de restructurer les données en JSON exploitable, pas de reproduire le document.

STRUCTURE JSON OBLIGATOIRE (un seul objet consolidé pour le batch) :

{
  "metadata": {
    "assureur": "",
    "nom_produit": "",
    "reference_contrat": "",
    "version": "",
    "pages_analysees": []
  },
  "preambule": "",
  "definitions": [{"terme": "", "definition": ""}],
  "garanties": [
    {
      "nom_garantie": "",
      "description": "",
      "conditions_activation": "",
      "exclusions_specifiques": [],
      "plafonds": [{"type": "", "montant": "", "conditions": ""}],
      "franchises": [{"type": "", "montant": "", "conditions": ""}],
      "tableau_associe": null
    }
  ],
  "exclusions_generales": [],
  "tableaux_reconstruits": [
    {
      "titre": "",
      "page": 0,
      "headers": [],
      "rows": [],
      "notes_pied": []
    }
  ],
  "images_detectees": [
    {"page": 0, "type": "logo | tampon | signature | pictogramme", "description": ""}
  ]
}

RÈGLES STRICTES :
- Utilise IMPÉRATIVEMENT les coordonnées (boxes) du layout JSON fourni pour reconstruire les tableaux avec précision.
- Un tableau = headers + rows avec mapping exact (jamais en texte brut).
- Gère les en-têtes multilignes en les fusionnant logiquement.
- Reformule les descriptions, extrait seulement les données clés.
- Si champ absent → "" ou [].
- Retourne UNIQUEMENT du JSON valide, sans texte supplémentaire ni ```json.

Analyse les pages et layouts fournis.
"""
    
    def process_batch_with_key_rotation(self, batch_items: List[tuple]) -> Optional[Dict[str, Any]]:
        """Process a batch of images with API key rotation"""
        contents = [self.get_system_prompt()]
        
        for img_name, layout_path in batch_items:
            page_num = int(Path(img_name).stem)
            
            # Load layout data
            with open(layout_path, 'r', encoding='utf-8') as f:
                layout_data = json.load(f)
            
            # Resize image if needed
            img = self.resize_image_if_needed(os.path.join(self.image_dir, img_name))
            
            page_instruction = f"""
Page {page_num} — Document interne confidentiel.
Layout géométrique complet (utilise les boxes pour reconstruire précisément les tableaux) :
{json.dumps(layout_data, ensure_ascii=False)}

Extrais les données de cette page en intégrant au JSON global.
"""
            
            contents.extend([page_instruction, img])
        
        # Try with multiple API keys
        max_attempts = len(self.api_keys) * 3  
        for attempt in range(max_attempts):
            client = self.get_current_client()
            try:
                logger.info(f"Attempting with API key #{self.current_key_index + 1} (attempt {attempt + 1})")
                response = client.models.generate_content(
                    model=self.model_name,
                    contents=contents
                )
                text = response.text.strip()
                text = text.replace("```json", "").replace("```", "").strip()
                return json.loads(text)
            
            except Exception as e:
                error_msg = str(e).lower()
                if "quota" in error_msg or "429" in error_msg or "rate limit" in error_msg:
                    logger.warning(f"Quota exceeded on key {self.current_key_index + 1}, switching to next key")
                    self.switch_to_next_key()
                    time.sleep(30 + attempt * 20)
                elif "finish_reason" in error_msg and "4" in error_msg:
                    logger.warning("Copyright blocking detected, retrying after delay")
                    time.sleep(40 + attempt * 20)
                else:
                    logger.error(f"Non-quota error: {e}, retrying in 15s")
                    time.sleep(15)
        
        logger.error("All API keys exhausted or failed")
        return None
    
    def merge_extractions(self, extractions: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Merge partial extractions into final result with robust error handling"""
        if not extractions:
            return None
        
        logger.info("Merging final extractions...")
        
        # Limit the data size to prevent JSON corruption
        data_str = json.dumps(extractions, ensure_ascii=False)[:40000] 
        
        fusion_prompt = f"""
Fusionne ces JSON partiels en un document final cohérent.

IMPORTANT: Retourne UNIQUEMENT du JSON valide, aucun texte avant ou après.

RÈGLES :
- Métadonnées uniques
- Continuer les garanties coupées 
- Cumuler tableaux sans doublon
- Supprimer doublons exclusions/définitions
- Si trop de données, résumer mais garder structure

DONNÉES : {data_str}

JSON final uniquement:
"""
        
        # Try multiple times with different approaches
        max_attempts = len(self.api_keys) * 2
        for attempt in range(max_attempts):
            try:
                client = self.get_current_client()
                response = client.models.generate_content(
                    model=self.model_name,
                    contents=[fusion_prompt]
                )
                
                # Clean the response more robustly
                text = response.text.strip()
                text = self._clean_json_response(text)
                
                # Try to parse
                return json.loads(text)
                
            except json.JSONDecodeError as e:
                logger.warning(f"JSON decode error on attempt {attempt + 1}: {e}")
                
                # Try to fix common issues
                if "Unterminated string" in str(e):
                    try:
                        fixed_text = self._fix_unterminated_strings(text)
                        result = json.loads(fixed_text)
                        logger.info("Successfully fixed JSON string termination")
                        return result
                    except:
                        logger.warning("Could not fix unterminated strings")
                
                # Switch API key and retry
                self.switch_to_next_key()
                time.sleep(10)
                
            except Exception as e:
                logger.error(f"Fusion attempt {attempt + 1} failed: {e}")
                self.switch_to_next_key()
                time.sleep(15)
        
        logger.error("Final merge failed - creating fallback")
        return self._create_fallback_merge(extractions)
    
    def _clean_json_response(self, text: str) -> str:
        """Clean Gemini response to extract valid JSON"""
        # Remove markdown code blocks
        if "```json" in text:
            text = text.split("```json", 1)[1]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        
        text = text.strip()
        
        # Find JSON boundaries
        start_idx = text.find("{")
        end_idx = text.rfind("}")
        
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            text = text[start_idx:end_idx + 1]
        
        return text
    
    def _fix_unterminated_strings(self, json_text: str) -> str:
        """Try to fix unterminated string issues in JSON"""
        lines = json_text.split('\n')
        fixed_lines = []
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            quote_count = line.count('"')
            
            if quote_count % 2 == 1:
                if ':' in line and not stripped.endswith('",') and not stripped.endswith('"'):
                    
                    if stripped.endswith(','):
                        line = line.rstrip(',') + '",'
                    else:
                        line = line + '"'
                elif stripped.endswith('\\'):
                    
                    line = line[:-1] + '"'
            
            fixed_lines.append(line)
        
        return '\n'.join(fixed_lines)
    
    def _create_fallback_merge(self, extractions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create a simple fallback when AI merge fails"""
        logger.info("Creating fallback merge...")
        
        fallback = {
            "metadata": {
                "fusion_status": "fallback_mode",
                "total_batches": len(extractions),
                "extraction_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "note": "AI fusion failed - manual consolidation may be needed"
            },
            "partial_results": extractions,
            "statistics": self._calculate_statistics(extractions)
        }
        
        try:
            merged_garanties = []
            merged_tableaux = []
            merged_metadata = {}
            
            for extraction in extractions:
                if isinstance(extraction, dict):
                   
                    if "garanties" in extraction:
                        merged_garanties.extend(extraction.get("garanties", []))
                    
                    
                    if "tableaux_reconstruits" in extraction:
                        merged_tableaux.extend(extraction.get("tableaux_reconstruits", []))

                    if "metadata" in extraction and not merged_metadata:
                        merged_metadata = extraction.get("metadata", {})
            
            fallback["consolidated"] = {
                "garanties": merged_garanties,
                "tableaux_reconstruits": merged_tableaux,
                "metadata": merged_metadata
            }
            
        except Exception as e:
            logger.warning(f"Even fallback merge had issues: {e}")
        
        return fallback
    
    def _calculate_statistics(self, extractions: List[Dict[str, Any]]) -> Dict[str, int]:
        """Calculate statistics from extractions"""
        try:
            total_garanties = 0
            total_tableaux = 0
            total_definitions = 0
            
            for extraction in extractions:
                if isinstance(extraction, dict):
                    total_garanties += len(extraction.get("garanties", []))
                    total_tableaux += len(extraction.get("tableaux_reconstruits", []))
                    total_definitions += len(extraction.get("definitions", []))
            
            return {
                "total_garanties": total_garanties,
                "total_tableaux": total_tableaux,
                "total_definitions": total_definitions,
                "successful_batches": len(extractions)
            }
        except:
            return {"error": "Could not calculate statistics"}
    
    def extract_from_documents(self) -> bool:
        """Main extraction process"""
        # Get image files
        image_files = sorted([f for f in os.listdir(self.image_dir)
                             if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        if not image_files:
            logger.error("No images found in directory")
            return False
        
        all_extractions = []
        logger.info(f"Starting processing: {len(image_files)} pages (batch size: {self.batch_size})")
        
        # Process in batches
        for i in range(0, len(image_files), self.batch_size):
            batch = image_files[i:i + self.batch_size]
            batch_data = []
            
            # Prepare batch data
            for img_name in batch:
                base_name = Path(img_name).stem
                layout_path = os.path.join(self.layout_dir, f"{base_name}.json")
                if os.path.exists(layout_path):
                    batch_data.append((img_name, layout_path))
                else:
                    logger.warning(f"Layout missing for: {img_name}")
            
            if not batch_data:
                continue
            
            logger.info(f"Processing batch {i//self.batch_size + 1}: {len(batch_data)} pages")
            result = self.process_batch_with_key_rotation(batch_data)
            
            if result:
                all_extractions.append(result)
                logger.info("Batch extraction successful")
            else:
                logger.error("Batch failed definitively")
            
           
            time.sleep(15)
        
        
        if all_extractions:
            final_data = self.merge_extractions(all_extractions)
            
            if final_data:
                
                if final_data.get("metadata", {}).get("fusion_status") == "fallback_mode":
                    
                    fallback_path = os.path.join(os.path.dirname(self.final_file), "extraction_assurance_fallback.json")
                    with open(fallback_path, "w", encoding="utf-8") as f:
                        json.dump(final_data, f, indent=4, ensure_ascii=False)
                    logger.info(f"📄 Fallback result saved: {fallback_path}")
                    logger.info("⚠️  AI fusion failed but data is preserved - manual review recommended")
                    return True
                else:
                    
                    with open(self.final_file, "w", encoding="utf-8") as f:
                        json.dump(final_data, f, indent=4, ensure_ascii=False)
                    logger.info(f"✅ Final extraction saved: {self.final_file}")
                    return True
            else:
                logger.error("Final merge failed completely")
                # Save raw backup
                with open(self.backup_file, "w", encoding="utf-8") as f:
                    json.dump(all_extractions, f, indent=4, ensure_ascii=False)
                logger.info(f"💾 Partial results saved to {self.backup_file}")
                return False
        else:
            logger.error("No successful extractions")
            return False


def main():
    """Main function"""
    try:
        extractor = GeminiExtractor()
        success = extractor.extract_from_documents()
        
        if success:
            logger.info("Document extraction completed successfully!")
        else:
            logger.error("Document extraction failed!")
            exit(1)
            
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        exit(1)


if __name__ == "__main__":
    main()
