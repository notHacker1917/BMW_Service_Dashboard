"""Cluster label generation using TF-IDF and GPT."""
import pandas as pd
import numpy as np
from typing import Dict, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
import asyncio
import json
import litellm
from tenacity import retry, stop_after_attempt, wait_exponential
from app.utils.logger import logger
from app.utils.config import LLM_MODEL, LLM_MAX_TOKENS, LLM_TEMPERATURE

litellm.set_verbose = False


class ClusterLabeler:
    def __init__(self):
        self.model = LLM_MODEL
        self.labels = {}

    def extract_tfidf_keywords(self, texts: list, n_keywords: int = 5) -> list:
        """Extract top TF-IDF keywords from cluster texts."""
        try:
            vectorizer = TfidfVectorizer(max_features=100, stop_words="english", min_df=1)
            tfidf_matrix = vectorizer.fit_transform(texts)
            feature_names = vectorizer.get_feature_names_out()
            
            # Get mean TF-IDF scores
            mean_scores = tfidf_matrix.mean(axis=0).A1
            top_indices = np.argsort(mean_scores)[-n_keywords:][::-1]
            
            keywords = [feature_names[i] for i in top_indices]
            return keywords
        except Exception as e:
            logger.warning(f"TF-IDF extraction error: {e}")
            return []

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def generate_label_with_gpt(self, keywords: list, sample_complaints: list) -> Dict:
        """Generate cluster label using GPT."""
        
        system_prompt = """You are an expert motorcycle quality assurance engineer.
        Generate a concise, technical label for a group of related failure reports.
        Return ONLY valid JSON with no markdown or extra text.
        
        JSON format:
        {
            "label": "brief technical label (max 50 chars)",
            "root_component": "primary affected component",
            "recurring_symptom": "main symptom description",
            "failure_pattern": "overall failure pattern description"
        }
        """
        
        sample_text = "\n".join([f"- {c[:100]}" for c in sample_complaints[:3]])
        
        user_prompt = f"""Generate a label for a group of failures with these characteristics:
        
        Keywords: {', '.join(keywords)}
        
        Sample complaints:
        {sample_text}
        
        Return only the JSON object."""
        
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
            
            try:
                label_data = json.loads(content)
                return label_data
            except json.JSONDecodeError:
                logger.warning(f"Malformed label JSON: {content[:100]}")
                # Return default structure
                return {
                    "label": " ".join(keywords[:2]) if keywords else "Unknown Failure",
                    "root_component": "",
                    "recurring_symptom": "",
                    "failure_pattern": "",
                }
        except Exception as e:
            logger.error(f"GPT label generation error: {e}")
            return None

    async def label_clusters_async(
        self,
        df_clustered: pd.DataFrame,
        df_entities: pd.DataFrame,
        df_feedback: pd.DataFrame,
    ) -> pd.DataFrame:
        """Generate labels for all clusters asynchronously."""
        logger.info("Generating cluster labels...")
        
        cluster_ids = sorted(set(df_clustered["cluster_id"]) - {-1})
        labels_list = []
        
        for cluster_id in cluster_ids:
            # Get all feedback IDs in cluster
            cluster_feedback_ids = df_clustered[df_clustered["cluster_id"] == cluster_id]["feedback_id"].tolist()
            
            # Get feedback texts
            feedback_texts = df_feedback[df_feedback["feedback_id"].isin(cluster_feedback_ids)]["customer_feedback"].tolist()
            
            # Extract keywords
            keywords = self.extract_tfidf_keywords(feedback_texts)
            
            # Generate label
            label_data = await self.generate_label_with_gpt(keywords, feedback_texts)
            
            if label_data:
                # Get failure frequency
                failure_frequency = len(cluster_feedback_ids)
                
                # Get representative complaint
                representative = feedback_texts[0] if feedback_texts else ""
                
                labels_list.append({
                    "cluster_id": cluster_id,
                    "label": label_data.get("label", ""),
                    "root_component": label_data.get("root_component", ""),
                    "recurring_symptom": label_data.get("recurring_symptom", ""),
                    "failure_frequency": failure_frequency,
                    "representative_complaint": representative,
                    "confidence": 0.85,
                })
            
            await asyncio.sleep(0.5)  # Rate limiting
        
        df_labels = pd.DataFrame(labels_list)
        logger.info(f"Generated labels for {len(df_labels)} clusters")
        
        return df_labels


async def label_clusters(
    clustered_feedback_path: str,
    entities_path: str,
    feedback_path: str,
    output_path: str = None,
) -> str:
    """Generate labels for clusters."""
    logger.info("Loading cluster data...")
    
    df_clustered = pd.read_csv(clustered_feedback_path, encoding="utf-8")
    df_entities = pd.read_csv(entities_path, encoding="utf-8")
    df_feedback = pd.read_csv(feedback_path, encoding="utf-8")
    
    logger.info(f"Loaded {len(df_clustered)} clustered records, {len(df_entities)} entities")
    
    labeler = ClusterLabeler()
    df_labels = await labeler.label_clusters_async(df_clustered, df_entities, df_feedback)
    
    if output_path is None:
        p = Path(feedback_path)
        if "_cleaned" in p.name:
            output_path = str(p.parent / p.name.replace("_cleaned.csv", "_cluster_labels.csv"))
        else:
            output_path = str(p.parent / f"{p.stem}_cluster_labels.csv")
    
    df_labels.to_csv(output_path, index=False, encoding="utf-8")
    logger.info(f"Saved cluster labels to {output_path}")
    
    return str(output_path)
