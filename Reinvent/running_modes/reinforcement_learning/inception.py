import numpy as np
import pandas as pd
from typing import Tuple, List

from running_modes.configurations.reinforcement_learning.inception_configuration import InceptionConfiguration
from reinvent_chemistry.conversions import Conversions
from rdkit.Chem.Scaffolds import MurckoScaffold


class Inception:
    def __init__(self, configuration: InceptionConfiguration, scoring_function, prior):
        self.configuration = configuration
        self._chemistry = Conversions()
        self.memory: pd.DataFrame = pd.DataFrame(columns=['smiles', 'scaffolds', 'score', 'likelihood'])
        #self.memory: pd.DataFrame = pd.DataFrame(columns=['smiles', 'score', 'likelihood'])
        self._load_to_memory(scoring_function, prior, self.configuration.smiles)


    def _load_to_memory(self, scoring_function, prior, smiles):
        if len(smiles):
            standardized_and_nulls = [self._chemistry.convert_to_rdkit_smiles(smile) for smile in smiles]
            standardized = [smile for smile in standardized_and_nulls if smile is not None]
            self.evaluate_and_add(standardized, scoring_function, prior)

    def _purge_memory(self):
        #unique_df = self.memory.drop_duplicates(subset=["smiles"])
        #sorted_df = unique_df.sort_values('score', ascending=False)
        #self.memory = sorted_df.head(self.configuration.memory_size)
        sorted_df = self.memory.sort_values('score', ascending=False).dropna()
        #sorted_unique_df = sorted_df.drop_duplicates(subset=["scaffolds"], keep='first')
        grouped_df = sorted_df.groupby('scaffolds').head(10)
        self.memory = grouped_df.head(self.configuration.memory_size)

    def evaluate_and_add(self, smiles, scoring_function, prior):
        if len(smiles) > 0:
            scaffolds = [MurckoScaffold.MurckoScaffoldSmiles(smi) for smi in smiles]
            score = scoring_function.get_final_score(smiles)
            likelihood = prior.likelihood_smiles(smiles)
            df = pd.DataFrame({"smiles": smiles, "scaffolds": scaffolds, "score": score.total_score, "likelihood": -likelihood.detach().cpu().numpy()})
            #df = pd.DataFrame({"smiles": smiles, "score": score.total_score, "likelihood": -likelihood.detach().cpu().numpy()})
            self.memory = self.memory.append(df)
            self._purge_memory()

    def add(self, smiles, score, neg_likelihood):
        # NOTE: likelihood should be already negative
        if len(smiles) > 0:
            scaffolds = list()
            for smi in smiles:
                try:
                    scaffolds.append(MurckoScaffold.MurckoScaffoldSmiles(smi))
                except:
                    scaffolds.append(None)
            df = pd.DataFrame({"smiles": smiles, "scaffolds": scaffolds, "score": score, "likelihood": neg_likelihood.detach().cpu().numpy()})
            self.memory = self.memory.append(df)
            self._purge_memory()

    def sample(self) -> Tuple[List[str], np.array, np.array]:
        sample_size = min(len(self.memory), self.configuration.sample_size)
        if sample_size > 0:
            sampled = self.memory.sample(sample_size)
            smiles = sampled["smiles"].values
            scores = sampled["score"].values
            prior_likelihood = sampled["likelihood"].values
            return smiles, scores, prior_likelihood
        return [], [], []


