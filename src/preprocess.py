
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os


class Preprocessor:
    def __init__(self, input_path: str, output_path: str):
        self.input_path = input_path
        self.output_path = output_path
        self.df = None
        self.filtered_df = None
        self.relevant_products = [
            "Credit card",
            "Personal loan",
            "Buy Now, Pay Later (BNPL)",
            "Savings account",
            "Money transfer, virtual currency, or mobile wallet"
        ]

    def load_data(self):
        print(f"Loading data from {self.input_path}")
        self.df = pd.read_csv(self.input_path)
        print(f"Data loaded: {self.df.shape[0]} records")

    def explore_data(self):
        print("\n=== Basic Info ===")
        print(self.df.info())

        print("\n=== Missing Values ===")
        print(self.df.isnull().sum())

        print("\n=== Top Products ===")
        print(self.df['Product'].value_counts())

    def visualize_distributions(self):
        print("Generating product distribution bar chart...")
        product_counts = self.df['Product'].value_counts().head(10)
        plt.figure(figsize=(10, 6))
        sns.barplot(y=product_counts.index, x=product_counts.values)
        plt.title("Top 10 Products by Number of Complaints")
        plt.xlabel("Number of Complaints")
        plt.ylabel("Product")
        plt.tight_layout()
        plt.savefig("/content/product_distribution.png")
        plt.close()

        print("Generating complaint narrative length histogram...")
        self.df['narrative_length'] = self.df['Consumer complaint narrative'].astype(str).apply(lambda x: len(x.split()))
        plt.figure(figsize=(10, 6))
        sns.histplot(self.df['narrative_length'], bins=50, kde=False)
        plt.title("Distribution of Complaint Narrative Lengths")
        plt.xlabel("Narrative Word Count")
        plt.ylabel("Number of Complaints")
        plt.tight_layout()
        plt.savefig("/content/narrative_length_distribution.png")
        plt.close()

    def filter_and_clean_data(self):
        print("Filtering data for relevant products and non-empty narratives...")
        df_filtered = self.df[
            (self.df['Product'].isin(self.relevant_products)) &
            (self.df['Consumer complaint narrative'].notna())
        ].copy()

        print(f"Remaining records after filtering: {df_filtered.shape[0]}")

        print("Cleaning text narratives...")
        df_filtered['cleaned_narrative'] = df_filtered['Consumer complaint narrative'].apply(self.clean_text)

        self.filtered_df = df_filtered

    @staticmethod
    def clean_text(text: str) -> str:
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def save_cleaned_data(self):
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        print(f"Saving cleaned data to {self.output_path}")
        self.filtered_df.to_csv(self.output_path, index=False)
      