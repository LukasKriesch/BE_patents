import os
import re
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy import stats
from sentence_transformers import SentenceTransformer
from umap import UMAP
from bertopic import BERTopic
from bertopic.cluster import BaseCluster
from bertopic.representation import KeyBERTInspired
from sklearn.feature_extraction.text import CountVectorizer
import fast_hdbscan
from transformers import AutoTokenizer, AutoModelForCausalLM
import datamapplot

class Dimensionality:
    def __init__(self, reduced_embeddings):
        self.reduced_embeddings = reduced_embeddings

    def fit(self, X):
        return self

    def transform(self, X):
        return self.reduced_embeddings

def load_data(filepath):
    """Load patent data from a feather file."""
    try:
        return pd.read_feather(filepath)
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def preprocess_data(data):
    """Preprocess the data to get unique abstracts."""
    return list(set(data["appln_abstract"]))

def create_stopwords():
    """Create a list of stopwords."""
    sw = CountVectorizer(stop_words='english').get_stop_words()
    sw = list(sw)
    additional_sw = ["utility", "model", "discloses", "provides", "comprises", "invention",
                     "provide", "solution", "jpo", "copyright", "solved"]
    sw.extend(additional_sw)
    return sw

def generate_topic_label(documents, keywords, tokenizer, model):
    """Generate a human-readable label for a topic using LLM."""
    prompt = f"""I have a topic that contains the following documents: {documents}. 
                 The topic is described by the following keywords: {keywords}. 
                 Based on the above information, can you give a broad and generic short label of the topic of at most 5 words?"""
    input_ids = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**input_ids, max_length=3000)
    outputs = tokenizer.decode(outputs[0])
    outputs = re.findall(r'\*\*(.*?)\*\*', outputs)
    return outputs[-1] if outputs else ""

def main(data_filepath):
    # Load data
    data = load_data(data_filepath)
    if data is None:
        return

    # Preprocess data
    docs = preprocess_data(data)

    # Initialize models
    embedding_model = SentenceTransformer('mixedbread-ai/mxbai-embed-large-v1')
    pat_embeddings = embedding_model.encode(docs)

    umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42)
    umap_embeddings = umap_model.fit_transform(pat_embeddings)
    umap_model = Dimensionality(umap_embeddings)

    clusterer = fast_hdbscan.HDBSCAN(min_cluster_size=5048)
    cluster_labels = clusterer.fit_predict(umap_embeddings)
    hdbscan_model = BaseCluster()

    stopwords = create_stopwords()
    vectorizer_model = CountVectorizer(ngram_range=(1, 2), stop_words=stopwords)
    representation_model = {"KeyBERT": KeyBERTInspired()}

    # Fit BERTopic model
    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        representation_model=representation_model,
        verbose=True,
    ).fit(docs, embeddings=pat_embeddings, y=cluster_labels)

    # Get topic info and reduce outliers
    t = topic_model.get_topic_info()
    topics = topic_model.topics_
    new_topics = topic_model.reduce_outliers(docs, topics)

    new_topics_grouped = pd.DataFrame(new_topics)
    data["topic"] = new_topics
    new_topics_grouped.columns = ["topic"]
    new_topics_grouped = new_topics_grouped.groupby("topic").size().reset_index()
    t = pd.merge(t, new_topics_grouped, left_on="Topic", right_on="topic", how="left")
    t["share"] = t[0] / len(docs)

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

    # Generate labels
    labels = []
    for topic in tqdm(list(t["Topic"])):
        test = data[data["topic"] == topic]
        test = test.sample(10)
        test_titles = list(test["appln_title"])
        keywords = list(t[t["Topic"] == topic]["KeyBERT"])
        labels.append(generate_topic_label(documents=test_titles, keywords=keywords, tokenizer=tokenizer, model=model))

    # Dimensionality reduction for plotting
    umap_model_2d = UMAP(n_neighbors=15, n_components=2, min_dist=0.0, metric='cosine', random_state=42)
    reduced_embeddings_2d = umap_model_2d.fit_transform(pat_embeddings)

    t["LLM"] = labels
    data["LLM"] = pd.merge(data, t, left_on="topic", right_on="Topic", how="left")
    data["embeddings_2d"] = list(reduced_embeddings_2d)
    data = data[data["topic"] < 36]
    all_labels = np.asarray(list(data["LLM"]))

    # Plot the data
    datamapplot.create_plot(
        reduced_embeddings_2d,
        all_labels,
        label_font_size=11,
        title="Bioeconomy Patents",
        sub_title="Topics labeled with `Meta-LLAMA-3-8B`",
        label_wrap_width=16,
        use_medoids=True,
        label_linespacing=1.25,
        dynamic_label_size=True,
        dpi=300,
        noise_label="Water Treatment and Filtration Systems",
        highlight_labels=["Organic Plant Cultivation Methods", "Animal Husbandry Feeding Mechanisms",
                          "Food Processing and Packaging Technology", "Aquatic Farming and Aquaculture Methods",
                          "Animal Feed and Nutrition"]
    )

if __name__ == "__main__":
    data_filepath = "BE_patents.feather"
    main(data_filepath)

