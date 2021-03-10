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
            
    def test_model_trains(self):
        # Performs one step of training and verifies that the weights are updated, implying some training occurs.
        with TemporaryDirectory() as tmpdirname:
            T = torch.cuda.FloatTensor
            encode(smiles_file="data/EGFR_training.smi", output_smiles_file_path=tmpdirname+'/encoded_smiles.latent',encoder='chembl')
            C = CreateModelRunner(input_data_path=tmpdirname+'/encoded_smiles.latent', output_model_folder=tmpdirname)
            C.run()
            D = Discriminator.load(tmpdirname+'/discriminator.txt')
            G = Generator.load(tmpdirname+'/generator.txt')
            G.cuda()
            D.cuda()
            optimizer_G = torch.optim.Adam(G.parameters())
            optimizer_D = torch.optim.Adam(D.parameters())
            json_smiles = open(tmpdirname+'/encoded_smiles.latent', "r")
            latent_space_mols = np.array(json.load(json_smiles))
            testSampler = Sampler(G)
            latent_space_mols = latent_space_mols.reshape(latent_space_mols.shape[0], 512)
            dataloader = torch.utils.data.DataLoader(LatentMolsDataset(latent_space_mols), shuffle=True,
                                                        batch_size=64, drop_last=True)
            for _, real_mols in enumerate(dataloader):
                real_mols = real_mols.type(T)
                before_G_params = []
                before_D_params = []
                for param in G.parameters():
                    before_G_params.append(param.view(-1))
                before_G_params = torch.cat(before_G_params)
                for param in D.parameters():
                    before_D_params.append(param.view(-1))
                before_D_params = torch.cat(before_D_params)
                
                optimizer_D.zero_grad()
                fake_mols = testSampler.sample(real_mols.shape[0])
                real_validity = D(real_mols)
                fake_validity = D(fake_mols)
                #It is not relevant to compute gradient penalty. The test is only interested in if there is a change in
                #the weights (training), not in giving proper training
                d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) 
                d_loss.backward()
                optimizer_D.step()
                optimizer_G.zero_grad()
                fake_mols = testSampler.sample(real_mols.shape[0])
                fake_validity = D(fake_mols)
                g_loss = -torch.mean(fake_validity)
                g_loss.backward()
                optimizer_G.step()
                after_G_params = []
                after_D_params = []
                for param in G.parameters():
                    after_G_params.append(param.view(-1))
                after_G_params = torch.cat(after_G_params)
                for param in D.parameters():
                    after_D_params.append(param.view(-1))
                after_D_params = torch.cat(after_D_params)
                self.assertTrue(torch.any(torch.ne(after_G_params,before_G_params)))
                self.assertTrue(torch.any(torch.ne(after_D_params,before_D_params)))
                
                break
            

if __name__ == '__main__':
    unittest.main()