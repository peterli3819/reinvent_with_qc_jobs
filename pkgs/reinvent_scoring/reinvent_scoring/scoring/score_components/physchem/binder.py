from reinvent_scoring.scoring.component_parameters import ComponentParameters
from reinvent_scoring.scoring.score_components.physchem.base_physchem_component import BasePhysChemComponent
from rdkit import Chem
from alfabet import model

class Binder(BasePhysChemComponent):
    def __init__(self, parameters: ComponentParameters):
        super().__init__(parameters)
        self.patt1 = Chem.MolFromSmarts("[#6]")
        self.patt2 = Chem.MolFromSmarts("[C]=[C]")
        self.patt3 = Chem.MolFromSmarts("[#7]")
        self.patt4 = Chem.MolFromSmarts("NC=O")
        self.patt5 = Chem.MolFromSmarts("[#6]OC(C=[CH2])=O")
        #self.patt6 = Chem.MolFromSmarts("[$([*]-[N,OH1])]")

    def _calculate_phys_chem_property(self, mol):
        # Check if number of carbon <= 12
        if len(mol.GetSubstructMatches(self.patt1)) > 12:
            return 0
        # Check if number of C=C double bond == 1
        elif len(mol.GetSubstructMatches(self.patt2)) != 1:
            return 0
        # Check if any N atom exists (besides -NHC=O and C(=O)NHC(=O))
        elif len(mol.GetSubstructMatches(self.patt3)) != len(set([atoms[0] for atoms in mol.GetSubstructMatches(self.patt4)])):
            return 0
        # Check if acrlyate exists
        elif len(mol.GetSubstructMatches(self.patt5)) != 1:
            return 0
        # Check if aliphatic N or O exists
        #elif len(mol.GetSubstructMatches(self.patt6)) == 0:
        #    return 0
        # Check if minimum BDFE for C-H bond larger than 87.2 kcal/mol
        smiles = Chem.MolToSmiles(mol)
        df = model.predict([smiles])
        min_bdfe = df[df['bond_type'] == 'C-H']['bdfe_pred'].min()
        #if min_bdfe < 87.2:
        #    return 0
        return min_bdfe

