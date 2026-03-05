from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class DiseaseDomainClassifier:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        self.domains = {
            "Neurodegenerative disorder": "neurodegenerative",
            "Genetic epileptic encephalopathy": "genetic_epileptic",
            "Neuroinflammatory condition": "neuroinflammatory",
            "Metabolic encephalopathy": "metabolic",
            "Structural brain malformation": "structural",
            "Vascular neurological injury": "vascular",
            "Demyelinating disorder": "demyelinating"
        }

        self.domain_texts = list(self.domains.keys())
        self.domain_embeddings = self.model.encode(self.domain_texts)

    def classify(self, disease_name):

        if not disease_name.strip():
            return None

        disease_embedding = self.model.encode([disease_name])

        similarities = cosine_similarity(disease_embedding, self.domain_embeddings)[0]

        best_index = int(np.argmax(similarities))

        return self.domains[self.domain_texts[best_index]]