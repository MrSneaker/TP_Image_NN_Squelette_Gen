import sys
import numpy as np
import cv2
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image
from GenVanillaNNImage import SkeToImageTransform
from VideoSkeleton import VideoSkeleton
from VideoReader import VideoReader
from Skeleton import Skeleton
from GenVanillaNNImage import GenNNSkeImToImage  # Assuming GenNNSkeToImage is the generator network

# Define the Discriminator class
class Discriminator(nn.Module):
    def __init__(self, ngpu=0):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1024, 4, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(1024, 1, 4, stride=1, padding=0), 
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.model(input)
    
    
class VideoSkeletonDataset(Dataset):
    def __init__(self, videoSke, ske_reduced, source_transform=None, target_transform=None):
        """ videoSkeleton dataset: 
                videoske(VideoSkeleton): video skeleton that associate a video and a skeleton for each frame
                ske_reduced(bool): use reduced skeleton (13 joints x 2 dim=26) or not (33 joints x 3 dim = 99)
        """
        self.videoSke = videoSke
        self.source_transform = source_transform
        self.target_transform = target_transform
        self.ske_reduced = ske_reduced
        self.ske_to_image = SkeToImageTransform(image_size=128)
        print("VideoSkeletonDataset: ",
              "ske_reduced=", ske_reduced, "=(", Skeleton.reduced_dim, " or ",Skeleton.full_dim,")" )


    def __len__(self):
        return self.videoSke.skeCount()


    def __getitem__(self, idx):
        ske = self.videoSke.ske[idx]
        stick_image = self.ske_to_image(ske)

        if self.source_transform:
            stick_image = self.source_transform(Image.fromarray(stick_image))

        image = Image.open(self.videoSke.imagePath(idx))
        if self.target_transform:
            image = self.target_transform(image)
            
                
        return stick_image, image
    
    
    def preprocessSkeleton(self, ske, source_transform=None):
        if source_transform:
            self.source_transform = source_transform
        if self.source_transform:
            ske = self.source_transform(ske)
        else:
            ske = torch.from_numpy( ske.__array__(reduced=self.ske_reduced).flatten() )
            ske = ske.to(torch.float32)
            ske = ske.reshape( ske.shape[0],1,1)
        return ske


    def tensor2image(self, normalized_image):
        numpy_image = normalized_image.detach().numpy()
        # Réorganiser les dimensions (C, H, W) en (H, W, C)
        numpy_image = np.transpose(numpy_image, (1, 2, 0))
        # passage a des images cv2 pour affichage
        numpy_image = cv2.cvtColor(np.array(numpy_image), cv2.COLOR_RGB2BGR)
        denormalized_image = numpy_image * np.array([0.5, 0.5, 0.5]) + np.array([0.5, 0.5, 0.5])
        denormalized_output = denormalized_image * 1
        return denormalized_output

# Define the GenGAN class
class GenGAN():
    """ Class that generates an image from a skeleton posture using a GAN """
    def __init__(self, videoSke, loadFromFile=False, batch_size=16):
        self.netG = GenNNSkeImToImage()  # Initialize generator
        self.netD = Discriminator()    # Initialize discriminator
        self.real_label = 1.
        self.fake_label = 0.
        self.filename = 'data/DanceGenGAN.pth'
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.netD.to(self.device)
        self.netG.to(self.device)
        
        image_size = 128
        tgt_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        
        self.dataset = VideoSkeletonDataset(videoSke, ske_reduced=True, target_transform=tgt_transform)
        self.dataloader = DataLoader(dataset=self.dataset, batch_size=batch_size, shuffle=True)

        if loadFromFile and os.path.isfile(self.filename):
            print("GenGAN: Loading from file:", self.filename)
            self.netG.load_state_dict(torch.load(self.filename))

        
        self.criterion = nn.MSELoss()
        self.optimizerD = optim.Adam(self.netD.parameters(), lr=0.0001, betas=(0.5, 0.999))
        self.optimizerG = optim.Adam(self.netG.parameters(), lr=0.0002, betas=(0.5, 0.999))

    def train(self, n_epochs=20):
        """ Training loop for GAN """
        try : 
            for epoch in range(n_epochs):
                epoch_loss = 0.0
                nb_sample = 0

                for i, (skeleton, real_images) in enumerate(self.dataloader):
                    real_labels = torch.full((real_images.size(0),), self.real_label * 0.9).to(self.device)  # Smooth labels
                    fake_labels = torch.full((real_images.size(0),), self.fake_label).to(self.device)

                    skeleton = skeleton.to(self.device)
                    real_images = real_images.to(self.device)

                    # Mise à jour du Discriminateur (1 fois sur 2)
                    if i % 2 == 0:  # Condition pour mettre à jour le discriminateur
                        self.netD.zero_grad()

                        # Real images
                        real_images = real_images.to(torch.float32)
                        output_real = self.netD(real_images).view(-1)
                        lossD_real = self.criterion(output_real, real_labels)
                        lossD_real.backward()

                        # Fake images
                        fake_images = self.netG(skeleton).detach()  # Détacher le graphe ici
                        output_fake = self.netD(fake_images).view(-1)
                        lossD_fake = self.criterion(output_fake, fake_labels)
                        lossD_fake.backward()

                        self.optimizerD.step()

                    # Mise à jour du Générateur (à chaque itération)
                    self.netG.zero_grad()
                    fake_images = self.netG(skeleton)  # Recréez les images factices
                    output_fake = self.netD(fake_images).view(-1)
                    lossG = self.criterion(output_fake, real_labels)
                    lossG.backward()
                    self.optimizerG.step()

                    epoch_loss += lossG.item()
                    nb_sample += 1

                    if i % 50 == 0:
                        print(f"Epoch {epoch+1}/{n_epochs}, Iter {i}, LossG = {lossG.item()}", end='\r')

                print(f"Epoch {epoch+1}/{n_epochs}, Avg LossG: {epoch_loss/nb_sample}", end='\r')

                if epoch % 10 == 0:
                    torch.save(self.netG.state_dict(), self.filename)
                    
        except KeyboardInterrupt:
            print('Training was interupted, saving..')
            torch.save(self.netG.state_dict(), self.filename)

    def generate(self, ske):
        """ Generate an image from a skeleton input """
        self.netG.eval()
        with torch.no_grad():
            ske_t = self.dataset.preprocessSkeleton(ske, SkeToImageTransform(image_size=128))
            ske_t = ske_t.to(self.device)
            ske_t_batch = ske_t.unsqueeze(0)        # make a batch
            normalized_output = self.netG(ske_t_batch)
            normalized_output = torch.Tensor.cpu(normalized_output)
            res = self.dataset.tensor2image(normalized_output[0])       # get image 0 from the batch
        return res


if __name__ == '__main__':
    force = False
    
    n_epoch = int(sys.argv[1])
    train = True
    if len(sys.argv) > 2:
        test_opcv = sys.argv[2] == '--test'
    else:
        test_opcv = False
    
    if len(sys.argv) > 3:
        batch_size = int(sys.argv[3])
    else:
        batch_size = 16
    
    if len(sys.argv) > 4:
        filename = sys.argv[4]
        if len(sys.argv) > 5:
            force = sys.argv[5].lower() == "true"
    else:
        filename = "data/taichi1.mp4"
    print("GenGAN: Current Working Directory=", os.getcwd())
    print("GenGAN: Filename=", filename)

    targetVideoSke = VideoSkeleton(filename)

    gen = GenGAN(targetVideoSke, loadFromFile=False)
    gen.train(n_epochs=n_epoch)
    
    for i in range(targetVideoSke.skeCount()):
        image = gen.generate(targetVideoSke.ske[i])
        resized_image = cv2.resize(image, (256, 256))
        cv2.imshow('Generated Image', resized_image)
        cv2.waitKey(0) 
    cv2.destroyAllWindows()
