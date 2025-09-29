import os.path
import pickle
import re

import nltk
import pandas as pd
from nltk.corpus import wordnet31
from transformers import AutoTokenizer


class Gelpe:

    def __init__(self, model, dataset, dataset_name):
        # self.tokenizer = AutoTokenizer.from_pretrained(model.model_served_name)
        nltk.download("wordnet", quiet=True)
        nltk.download("wordnet31", quiet=True)
        self.lemmatizer = wordnet31

        self.name = "gelpe"
        self.local_expl = "lime"
        self.aggr = "avg_50"
        self.global_expl = "cart"
        self.surrogate_path = (
            "outs/global_explainers/theories/predictors/{}_{}_{}_{}_{}.pkl".format(
                model.name, self.local_expl, self.global_expl, self.aggr, dataset_name
            )
        )

        if os.path.exists(self.surrogate_path):
            with open(self.surrogate_path, "rb") as file:
                self.explainer = pickle.load(file)
                self.loaded = True
        # else:
        #     self.explainer = Explainer(
        #         model=model,
        #         dataset=dataset,
        #         device="cpu",
        #         train_dataset=dataset_name,
        #         test_dataset=dataset_name,
        #         local_exp_file_available=False,
        #         loc_explain_mode=self.local_expl,
        #         aggregator=self.aggr,
        #         global_explainer=self.global_expl,
        #         max_skipgrams_length=2,
        #         run_mode="glob",
        #         logger=None,
        #         out_folder="outs/".format(dataset_name),
        #     )
        #     self.loaded = False

    def run(self):
        if not self.loaded:
            self.explainer.run()

    def get_expl(self):
        pass

    def predict(self, x):
        text_label = self.predict_with_cart(self.explainer, x, self.lemmatizer)[0]
        return text_label == "unsafe"

    def predict_with_cart(self, cart, sentence, lemmatizer):
        valuable_tokens = cart.get_valuable_tokens()

        dataframe = pd.DataFrame(index=[0], columns=valuable_tokens)
        dataframe = self.convert_sentence_to_dataframe_row(
            sentence=sentence,
            lemmatizer=lemmatizer,
            dataframe=dataframe,
            valuable_tokens=valuable_tokens,
            index=0,
        )
        prediction = cart.predict(dataframe)

        return prediction

    def convert_sentence_to_dataframe_row(
        self,
        sentence: str,
        lemmatizer,
        dataframe: pd.DataFrame,
        valuable_tokens,
        index: int = 0,
    ):
        # Lemmatize words in order to match correctly the relevant jargons
        words = [
            lemmatizer.morphy(word) if lemmatizer.morphy(word) is not None else word
            for word in re.split(r"\W+", sentence)
        ]
        # Construct list of skipgrams in the sentence that is at hand
        print(words)

        all_sequences_in_sentence = self.skipgrams_from_sentence(words, max_length=2)
        all_sequences_in_sentence = [
            " + ".join(t) if isinstance(t, tuple) else t
            for t in all_sequences_in_sentence
        ]

        skip = "Ġ"
        all_sequences_in_sentence = [
            s.replace(skip, "") for s in all_sequences_in_sentence
        ]
        print(all_sequences_in_sentence)

        # Construct an empty row which will be added to the dataframe
        row = [1 if seq in all_sequences_in_sentence else 0 for seq in valuable_tokens]
        dataframe.iloc[index] = {
            valuable_tokens[i]: row[i] for i in range(len(valuable_tokens))
        }
        return dataframe

    def skipgrams_from_sentence(self, words, max_length=3):
        # Construct list of skipgrams in the sentence that is at hand
        all_sequences_in_sentence = list(words.copy())
        max_length = max_length
        for length in range(2, max_length + 1):
            sequences_in_sentence = [
                tup
                for tup in list(nltk.skipgrams(words, length, 2))
                if all([isinstance(item, str) for item in tup])
            ]
            all_sequences_in_sentence += sequences_in_sentence
        return all_sequences_in_sentence
