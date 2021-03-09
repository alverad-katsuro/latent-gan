from autoencoder import autoencoder
from rdkit import Chem
import numpy as np
import unittest


class test_heteroencoder(unittest.TestCase):
    def test_decoder(self):
        self.model = autoencoder.load_model(model_version='chembl')
        test_aspirin = 'CC(=O)OC1=CC=CC=C1C(=O)O'
        # The decoder is deterministic and should always generate the same output
        binary_mol = [Chem.rdchem.Mol.ToBinary(Chem.MolFromSmiles(test_aspirin))]
        latent = self.model.transform(self.model.vectorize(binary_mol))
        latent=latent.squeeze(0)
        first, _ = self.model.predict(latent, temp=0)
        second, _= self.model.predict(latent, temp=0)
        self.assertEqual(first,second, 
                        "Model has encoded the same latent vector as two different SMILES")
       
            

if __name__ == '__main__':
    unittest.main()
