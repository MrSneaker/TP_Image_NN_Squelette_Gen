import sys
import numpy as np
import cv2
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from GenVanillaNNImage import VideoSkeletonDataset
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

# Define the GenGAN class
class GenGAN():
    """ Class that generates an image from a skeleton posture using a GAN """
    def __init__(self, videoSke, loadFromFile=False, batch_size=16):
        self.netG = GenNNSkeImToImage()  # Initialize generator
        self.netD = Discriminator()    # Initialize discriminator
        self.real_label = 1.
        self.fake_label = 0.
        self.filename = 'data/DanceGenGAN.pth'
        
        
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
            self.netG = torch.load(self.filename)

        self.criterion = nn.BCELoss()
        self.optimizerD = optim.Adam(self.netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.optimizerG = optim.Adam(self.netG.parameters(), lr=0.0002, betas=(0.5, 0.999))

    def train(self, n_epochs=20):
        """ Training loop for GAN """
        for epoch in range(n_epochs):
            for i, (skeleton, real_images) in enumerate(self.dataloader):
                real_labels = torch.full((real_images.size(0),), self.real_label) * 0.9  # Smooth label
                fake_labels = torch.full((real_images.size(0),), self.fake_label)

                self.netD.zero_grad()
                
                real_images = real_images.to(torch.float32)
                output_real = self.netD(real_images).view(-1)
                lossD_real = self.criterion(output_real, real_labels)
                lossD_real.backward()
                
                fake_images = self.netG(skeleton)
                output_fake = self.netD(fake_images.detach()).view(-1)
                lossD_fake = self.criterion(output_fake, fake_labels)
                lossD_fake.backward()
                self.optimizerD.step()

                self.netG.zero_grad()
                output_fake = self.netD(fake_images).view(-1)
                lossG = self.criterion(output_fake, real_labels)
                lossG.backward()
                self.optimizerG.step()

                if i % 50 == 0:
                    print(f"[{epoch}/{n_epochs}][{i}/{len(self.dataloader)}] Loss_D: {lossD_real+lossD_fake:.4f} Loss_G: {lossG:.4f}")

    def generate(self, ske):
        """ Generate an image from a skeleton input """
        ske_t = torch.from_numpy(ske.__array__(reduced=True).flatten()).to(torch.float32)
        ske_t = ske_t.reshape(1, Skeleton.reduced_dim, 1, 1)  
        self.netG.eval()
        with torch.no_grad():
            generated_image = self.netG(ske_t)
        generated_image = self.dataset.tensor2image(generated_image[0])
        return generated_image

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
    
    if len(sys.argv) > 3:
        filename = sys.argv[3]
        if len(sys.argv) > 4:
            force = sys.argv[4].lower() == "true"
    else:
        filename = "data/taichi1.mp4"
    print("GenGAN: Current Working Directory=", os.getcwd())
    print("GenGAN: Filename=", filename)

    targetVideoSke = VideoSkeleton(filename)

    gen = GenGAN(targetVideoSke, loadFromFile=False)
    gen.train(n_epochs=4)
    
    for i in range(targetVideoSke.skeCount()):
        image = gen.generate(targetVideoSke.ske[i])
        resized_image = cv2.resize(image, (256, 256))
        cv2.imshow('Generated Image', resized_image)
        cv2.waitKey(0) 
    cv2.destroyAllWindows()
