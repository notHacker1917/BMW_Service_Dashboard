"""Format structured entities into semantic text for embedding."""
import pandas as pd
from app.utils.logger import logger


def format_entities_for_embedding(df: pd.DataFrame) -> pd.DataFrame:
    """Convert structured entities into semantic text."""
    logger.info(f"Formatting {len(df)} entity records for embedding...")
    
    formatted_texts = []
    
    for _, row in df.iterrows():
        text = (
            f"Component: {row.get('component', '')} | "
            f"Failure: {row.get('failure_mode', '')} | "
            f"Symptom: {row.get('symptom', '')} | "
            f"Severity: {row.get('severity', '')} | "
            f"Condition: {row.get('driving_condition', '')}"
        )
        formatted_texts.append(text)
    
    df["formatted_text"] = formatted_texts
    logger.info(f"Formatted {len(df)} records successfully")
    
    return df


def format_from_file(input_path: str, output_path: str = None) -> str:
    """Format entities from CSV file."""
    logger.info(f"Loading entities from {input_path}")
    
    df = pd.read_csv(input_path, encoding="utf-8")
    df = format_entities_for_embedding(df)
    
    if output_path is None:
        output_path = input_path.replace("_entities", "_formatted")
    
    df.to_csv(output_path, index=False, encoding="utf-8")
    logger.info(f"Saved formatted entities to {output_path}")
    
    return output_path
