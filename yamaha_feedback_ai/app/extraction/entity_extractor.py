"""GPT-based structured entity extraction."""
import json
import asyncio
import pandas as pd
from typing import Dict, List, Optional
import aiohttp
from tenacity import retry, stop_after_attempt, wait_exponential
from app.utils.logger import logger
from app.utils.config import LLM_MODEL, LLM_MAX_TOKENS, LLM_TEMPERATURE
import litellm

litellm.set_verbose = False


class EntityExtractor:
    def __init__(self, model: str = LLM_MODEL):
        self.model = model
        self.failed_rows = []
        self.success_count = 0
        self.malformed_count = 0

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def extract_entities(self, feedback: str, language: str = "en") -> Optional[Dict]:
        """Extract structured entities from feedback using GPT."""
        
        system_prompt = """You are an expert motorcycle quality assurance engineer. 
        Extract structured failure information from customer feedback. 
        Return ONLY valid JSON with no markdown, no extra text.
        
        JSON format:
        {
            "component": "affected part (e.g., display, engine, clutch)",
            "failure_mode": "type of failure (e.g., freezing, overheating, disconnection)",
            "symptom": "observable behavior (e.g., navigation crash)",
            "severity": "critical|high|medium|low",
            "driving_condition": "conditions when issue occurs",
            "sentiment": "negative|neutral|positive"
        }
        """
        
        user_prompt = f"""Extract entities from this feedback (language: {language}):
        
        {feedback}
        
        Return only the JSON object, no additional text."""
        
        try:
            response = await asyncio.to_thread(
                litellm.completion,
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=LLM_MAX_TOKENS,
                temperature=LLM_TEMPERATURE,
            )
            
            content = response.choices[0].message.content.strip()
            
            # Try to parse JSON
            try:
                entities = json.loads(content)
                self.success_count += 1
                return self._validate_entities(entities)
            except json.JSONDecodeError:
                logger.warning(f"Malformed JSON response: {content[:100]}")
                self.malformed_count += 1
                return self._repair_json(content)
                
        except Exception as e:
            logger.error(f"Entity extraction error: {e}")
            self.failed_rows.append(feedback[:100])
            return None

    def _validate_entities(self, entities: Dict) -> Dict:
        """Validate and normalize extracted entities."""
        required_fields = ["component", "failure_mode", "symptom", "severity", "driving_condition", "sentiment"]
        
        # Ensure all fields exist
        for field in required_fields:
            if field not in entities:
                entities[field] = ""
        
        # Normalize severity
        severity = entities.get("severity", "").lower()
        valid_severities = ["critical", "high", "medium", "low"]
        if severity not in valid_severities:
            entities["severity"] = "medium"

        # Normalize sentiment
        sentiment = entities.get("sentiment", "").lower()
        if sentiment not in ["negative", "neutral", "positive"]:
            entities["sentiment"] = "negative"
        
        # Truncate long fields
        for key in entities:
            if isinstance(entities[key], str):
                entities[key] = entities[key][:200]
        
        return entities

    def _repair_json(self, content: str) -> Optional[Dict]:
        """Attempt to repair malformed JSON."""
        try:
            # Remove markdown code blocks
            content = content.replace("```json", "").replace("```", "").strip()
            
            # Find JSON object
            start = content.find("{")
            end = content.rfind("}") + 1
            
            if start >= 0 and end > start:
                json_str = content[start:end]
                entities = json.loads(json_str)
                return self._validate_entities(entities)
        except Exception as e:
            logger.warning(f"Failed to repair JSON: {e}")
        
        return None

    async def extract_batch(self, feedbacks: List[str], languages: List[str]) -> List[Dict]:
        """Extract entities from multiple feedbacks with rate limiting."""
        results = []
        batch_size = 5
        
        for i in range(0, len(feedbacks), batch_size):
            batch_feedbacks = feedbacks[i:i+batch_size]
            batch_languages = languages[i:i+batch_size]
            
            tasks = [
                self.extract_entities(fb, lang)
                for fb, lang in zip(batch_feedbacks, batch_languages)
            ]
            
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f"Batch processing error: {result}")
                    results.append(None)
                else:
                    results.append(result)
            
            logger.info(f"Processed batch {i//batch_size + 1}: {len([r for r in batch_results if r])} successful")
            await asyncio.sleep(1)  # Rate limiting
        
        return results

    async def extract_from_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract entities from DataFrame feedback column."""
        logger.info(f"Starting entity extraction on {len(df)} records...")
        
        feedbacks = df["customer_feedback"].tolist()
        languages = df["language"].tolist()
        
        # Run batch extraction
        results = await self.extract_batch(feedbacks, languages)
        
        # Convert to DataFrame
        entities_list = []
        for i, result in enumerate(results):
            if result:
                entities_list.append({
                    "feedback_id": df.iloc[i]["feedback_id"],
                    "component": result.get("component", ""),
                    "failure_mode": result.get("failure_mode", ""),
                    "symptom": result.get("symptom", ""),
                    "severity": result.get("severity", ""),
                    "driving_condition": result.get("driving_condition", ""),
                    "sentiment": result.get("sentiment", "negative"),
                    "confidence": 0.95 if result else 0.0,
                })
        
        df_entities = pd.DataFrame(entities_list)
        
        # Log statistics
        logger.info(f"Extraction complete: {self.success_count} successful, {self.malformed_count} malformed, {len(self.failed_rows)} failed")
        
        return df_entities


async def extract_entities_from_file(input_path: str, output_path: str = None) -> str:
    """Extract entities from cleaned feedback CSV."""
    logger.info(f"Loading cleaned feedback from {input_path}")
    
    df = pd.read_csv(input_path, encoding="utf-8")
    logger.info(f"Loaded {len(df)} records")
    
    extractor = EntityExtractor()
    df_entities = await extractor.extract_from_dataframe(df)
    
    if output_path is None:
        output_path = input_path.replace("processed/", "processed/").replace("_cleaned", "_entities")
    
    df_entities.to_csv(output_path, index=False, encoding="utf-8")
    logger.info(f"Saved extracted entities to {output_path}")
    
    return output_path
