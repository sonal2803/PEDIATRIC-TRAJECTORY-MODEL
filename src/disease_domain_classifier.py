from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

DEFAULT_DOMAIN = "structural"
CONFIDENCE_THRESHOLD = 0.25  # below this similarity, fall back to default


class DiseaseDomainClassifier:
    DOMAIN_MAP = {
        "Neurodegenerative disorder (e.g. Batten disease, Rett syndrome, Krabbe)":
            "neurodegenerative",
        "Genetic epileptic encephalopathy (e.g. Dravet syndrome, CDKL5, Lennox-Gastaut)":
            "genetic_epileptic",
        "Neuroinflammatory condition (e.g. autoimmune encephalitis, ADEM, MS)":
            "neuroinflammatory",
        "Metabolic encephalopathy (e.g. PKU, urea cycle disorder, organic acidaemia)":
            "metabolic",
        "Structural brain malformation (e.g. lissencephaly, polymicrogyria, holoprosencephaly)":
            "structural",
        "Vascular neurological injury (e.g. perinatal stroke, AVM, CSVT)":
            "vascular",
        "Demyelinating disorder (e.g. Pelizaeus-Merzbacher, leukodystrophy, NMO)":
            "demyelinating",
    }

    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.domain_texts = list(self.DOMAIN_MAP.keys())
        self.domain_keys  = list(self.DOMAIN_MAP.values())
        self.domain_embeddings = self.model.encode(self.domain_texts)

    def classify(self, disease_name: str) -> str:
        """
        Returns a domain string. Never returns None.

        Args:
            disease_name : free-text disease name from the user

        Returns:
            domain string (one of 7 classes, or DEFAULT_DOMAIN as fallback)
        """
        if not disease_name or not disease_name.strip():
            return DEFAULT_DOMAIN

        query_embedding = self.model.encode([disease_name.strip()])
        similarities = cosine_similarity(query_embedding, self.domain_embeddings)[0]
        best_index = int(np.argmax(similarities))
        best_score = float(similarities[best_index])

        if best_score < CONFIDENCE_THRESHOLD:
            return DEFAULT_DOMAIN

        return self.domain_keys[best_index]

    def classify_with_confidence(self, disease_name: str) -> tuple:
        """Returns (domain_str, confidence_score)."""
        if not disease_name or not disease_name.strip():
            return DEFAULT_DOMAIN, 0.0

        query_embedding = self.model.encode([disease_name.strip()])
        similarities = cosine_similarity(query_embedding, self.domain_embeddings)[0]
        best_index = int(np.argmax(similarities))
        return self.domain_keys[best_index], float(similarities[best_index])