from reinvent_scoring.scoring.component_parameters import ComponentParameters
from reinvent_scoring.scoring.score_components import BaseScoreComponent
from reinvent_scoring.scoring.score_summary import ComponentSummary
from rdkit import Chem
import numpy as np
from typing import List

from scscore.standalone_model_numpy import *

class SCScore(BaseScoreComponent):
    def __init__(self, parameters: ComponentParameters):
        super().__init__(parameters)
        model = SCScorer()
        self.scs_model = model.restore(os.path.join(project_root, 'models', 'full_reaxys_model_1024bool', 'model.ckpt-10654.as_numpy.json.gz'))

    def calculate_score(self, molecules: List) -> ComponentSummary:
        score = self._calculate_scs(molecules)
        score_summary = ComponentSummary(total_score=score, parameters=self.parameters)
        return score_summary

    def _calculate_scs(self, query_mols) -> np.array:
        scs_scores = []
        for mol in query_mols:
            try:
                smiles = Chem.MolToSmiles(mol)
                (smi, sco) = self.scs_model.get_score_from_smi(smiles)

            except:
                sco = 0.0
            scs_scores.append(sco)

        return np.array(scs_scores, dtype=np.float32)

