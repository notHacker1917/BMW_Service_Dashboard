"""Data cleaning and preprocessing."""
import pandas as pd
import re
from typing import Tuple
from app.utils.logger import logger

class DataCleaner:
    def __init__(self):
        self.removed_rows = {}
        self.normalization_stats = {}

    def clean_feedback(self, df: pd.DataFrame) -> pd.DataFrame:
        """Execute full data cleaning pipeline."""
        logger.info(f"Starting data cleaning on {len(df)} records...")
        
        initial_count = len(df)
        
        # Remove duplicates
        df = self._remove_duplicates(df)
        
        # Remove null rows
        df = self._remove_null_rows(df)
        
        # Clean text
        df["customer_feedback"] = df["customer_feedback"].apply(self._clean_text)
        
        final_count = len(df)
        removed_count = initial_count - final_count
        
        logger.info(f"Data cleaning complete: {removed_count} rows removed, {final_count} rows remaining")
        self.normalization_stats["initial_rows"] = initial_count
        self.normalization_stats["removed_rows"] = removed_count
        self.normalization_stats["final_rows"] = final_count
        
        return df

    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate feedback."""
        before = len(df)
        df = df.drop_duplicates(subset=["customer_feedback"], keep="first")
        after = len(df)
        removed = before - after
        self.removed_rows["duplicates"] = removed
        logger.info(f"Removed {removed} duplicate records")
        return df

    def _remove_null_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove rows with null values in critical columns."""
        before = len(df)
        critical_columns = ["feedback_id", "customer_feedback", "language", "vehicle_model"]
        df = df.dropna(subset=critical_columns)
        after = len(df)
        removed = before - after
        self.removed_rows["null_rows"] = removed
        logger.info(f"Removed {removed} null rows")
        return df

    def _clean_text(self, text: str) -> str:
        """Clean individual text record."""
        if not isinstance(text, str):
            return ""
        
        # Remove VIN patterns (17-character alphanumeric)
        text = re.sub(r'\b[A-HJ-NPR-Z0-9]{17}\b', '', text, flags=re.IGNORECASE)
        
        # Remove email addresses
        text = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '', text)
        
        # Remove phone numbers (various formats)
        text = re.sub(r'\b[\+]?[(]?[0-9]{3}[)]?[-\s\.]?[0-9]{3}[-\s\.]?[0-9]{4,6}\b', '', text)
        text = re.sub(r'[\+]?[0-9]{1,3}[-\s\.]?[0-9]{6,14}', '', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        # Ensure UTF-8 encoding
        text = text.encode('utf-8', 'ignore').decode('utf-8')
        
        return text

    def get_statistics(self) -> dict:
        """Get cleaning statistics."""
        return self.normalization_stats


def clean_raw_data(input_path: str, output_path: str = None) -> Tuple[pd.DataFrame, str]:
    """Load raw data, clean it, and save cleaned version."""
    logger.info(f"Loading raw data from {input_path}")
    
    df = pd.read_csv(input_path, encoding="utf-8")
    logger.info(f"Loaded {len(df)} records")
    
    cleaner = DataCleaner()
    df_clean = cleaner.clean_feedback(df)
    
    if output_path is None:
        output_path = input_path.replace("raw/", "processed/").replace(".csv", "_cleaned.csv")
    
    df_clean.to_csv(output_path, index=False, encoding="utf-8")
    logger.info(f"Saved cleaned data to {output_path}")
    
    # Log statistics
    stats = cleaner.get_statistics()
    logger.info(f"Cleaning stats: {stats}")
    
    return df_clean, output_path
