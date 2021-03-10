from ddc_pub import ddc_v3 as ddc
from rdkit import Chem
import numpy as np
import unittest
import os


class test_heteroencoder(unittest.TestCase):
    def setUp(self):
        path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'autoencoder/chembl_pretrained')
        self.model = ddc.DDC(model_name=path)
        self.reference_molecule= 'CC(=O)OC1=CC=CC=C1C(=O)O'
        self.binarization= [Chem.rdchem.Mol.ToBinary(Chem.MolFromSmiles(self.reference_molecule))]
        
    def test_decoder_deterministic(self):
        # The decoder is deterministic and should have no variance when decoding the same latent vector
        latent=np.random.rand(1,1,512)
        latent=latent.squeeze(0)
        first, _ = self.model.predict(latent, temp=0)
        second, _= self.model.predict(latent, temp=0)
        self.assertEqual(first,second, 
                        "Model has encoded the same latent vector as two different SMILES")
    
    def test_chembl_baseline_model(self):
        # The encoder/decoder pair should consistently have the same output for any specific input 
        # This test checks the robustness of the baseline chembl model to encoder noise.
        binary_mol = [Chem.rdchem.Mol.ToBinary(Chem.MolFromSmiles(self.reference_molecule))]
        latent = self.model.transform(self.model.vectorize(binary_mol))
        latent=latent.squeeze(0)
        smiles, _ = self.model.predict(latent, temp=0)
        # The same output molecule can have several corresponding smiles and the decoder is trained
        # to output one of these. This canonicalization verifies that indeed the same molecule is
        # produced every time
        canonical_form = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
        self.assertEqual(canonical_form,'CC(=O)Oc1ccccc1C(=O)O',
                            "Decoder output does not match expected output of ChEMBL trained heteroencoder")
            


if __name__ == '__main__':
    unittest.main()
