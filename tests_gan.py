from tempfile import TemporaryDirectory
from encode import encode
from runners.CreateModelRunner import CreateModelRunner
from models.Discriminator import Discriminator
from models.Generator import Generator
from runners.TrainModelRunner import TrainModelRunner
import torch
import unittest
import numpy as np
import json
from src.Sampler import Sampler
from datasets.LatentMolsDataset import LatentMolsDataset

class test_GAN(unittest.TestCase):
    
    def test_separate_optimizers(self):
        # Verify that two different instances of the optimizer is created using the TrainModelRunner.py initialization
        # This ensures the two components train separately 
        with TemporaryDirectory() as tmpdirname:
            encode(smiles_file="data/EGFR_training.smi", output_smiles_file_path=tmpdirname+'/encoded_smiles.latent',encoder='chembl')
            C = CreateModelRunner(input_data_path=tmpdirname+'/encoded_smiles.latent', output_model_folder=tmpdirname)
            C.run()
            D = Discriminator.load(tmpdirname+'/discriminator.txt')
            G = Generator.load(tmpdirname+'/generator.txt')
            optimizer_G = torch.optim.Adam(G.parameters())
            optimizer_D = torch.optim.Adam(D.parameters())
            self.assertTrue(type(optimizer_G) == type(optimizer_D))  # must return the same type of object
            self.assertTrue(optimizer_G is not optimizer_D)          # object identity MUST be different
            

            

if __name__ == '__main__':
    unittest.main()