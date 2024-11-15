import numpy as np
import cv2
import os
import pickle
import sys
import math

from PIL import Image
import matplotlib.pyplot as plt
from torchvision.io import read_image

import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset
from torchvision import transforms

#from tensorboardX import SummaryWriter

from VideoSkeleton import VideoSkeleton
from VideoReader import VideoReader
from Skeleton import Skeleton

torch.set_default_dtype(torch.float32)


class SkeToImageTransform:
    def __init__(self, image_size):
        self.imsize = image_size

    def __call__(self, ske):
        #image = Image.new('RGB', (self.imsize, self.imsize), (255, 255, 255))
        image = white_image = np.ones((self.imsize, self.imsize, 3), dtype=np.uint8) * 255
        Skeleton.draw_reduced(ske.reduce(), image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # cv2.imshow('Image', image)
        # key = cv2.waitKey(-1)
        image = np.transpose(image, (2, 0, 1))
        image = torch.from_numpy(image).float()
        return image



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

        # Apply source_transform if provided (e.g., normalization or augmentation)
        if self.source_transform:
            stick_image = self.source_transform(Image.fromarray(stick_image))

        # Load and transform the target image
        image = Image.open(self.videoSke.imagePath(idx))
        if self.target_transform:
            image = self.target_transform(image)
            
                
        return stick_image, image
    
    
    def preprocessSkeleton(self, ske):
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





def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)




class GenNNSkeImToImage(nn.Module):
    """ class that Generate a new image from videoSke from a new skeleton posture
       Fonc generator(Skeleton)->Image
    """
    def __init__(self):
        super(GenNNSkeImToImage, self).__init__()
        self.input_dim = Skeleton.reduced_dim
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.model = nn.Sequential(
            # TP-TODO
            nn.Conv2d(3, 256, kernel_size=4, stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            
            nn.Conv2d(256, 128, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            
            nn.Conv2d(128, 64, kernel_size=4, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            
            nn.Conv2d(64, 32, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            
            nn.Conv2d(32, 16, kernel_size=4, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            
            nn.Conv2d(16, 3, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(3),
            nn.LeakyReLU()
        )
        self.to(device=self.device)
        print(self.model)
        

    def forward(self, z):
        img = self.model(z)
        return img







class GenVanillaNNImage():
    """ class that Generate a new image from videoSke from a new skeleton posture
       Fonc generator(Skeleton)->Image
    """
    def __init__(self, videoSke, loadFromFile=False, batch_size=32):
        image_size = 128
        self.netG = GenNNSkeImToImage()
        src_transform = SkeToImageTransform(image_size)
        self.filename = 'data/DanceGenVanillaFromIm.pth'
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        tgt_transform = transforms.Compose([
                            transforms.Resize(image_size),
                            transforms.CenterCrop(image_size),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            # [transforms.Resize((64, 64)),
                            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                            ])
        self.dataset = VideoSkeletonDataset(videoSke, ske_reduced=True, target_transform=tgt_transform, source_transform=src_transform)
        self.dataloader = torch.utils.data.DataLoader(dataset=self.dataset, batch_size=batch_size, shuffle=True)
        if loadFromFile and os.path.isfile(self.filename):
            print("GenVanillaNNImage: Load=", self.filename)
            print("GenVanillaNNImage: Current Working Directory: ", os.getcwd())
            if torch.cuda.is_available():
                self.netG.load_state_dict(torch.load(self.filename))
            else:
                self.netG.load_state_dict(torch.load(self.filename, map_location=torch.device('cpu')))


    def train(self, n_epochs=20):
        # TP-TODO
        optimizer = torch.optim.Adam(self.netG.parameters(), lr=0.0001)
        criterion = nn.MSELoss()
        
        try:
            for n in range(n_epochs):
                epoch_loss = 0.0
                nb_sample = 0
                for x, t in self.dataloader:
                    optimizer.zero_grad()
                    x = x.to(self.device)
                    t = t.to(self.device)

                    out = self.netG(x)
                    loss = criterion(out, t)
                    
                    
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                    nb_sample += 1
                
                print(f"Epoch {n+1}/{n_epochs}, Loss: {epoch_loss/nb_sample}")
                if n % 100 == 0:
                    torch.save(self.netG.state_dict(), self.filename)
            torch.save(self.netG.state_dict(), self.filename)
        except KeyboardInterrupt:
            print("Train was interupted, saving..")
            torch.save(self.netG.state_dict(), self.filename)
            
            
            
                

    def generate(self, ske):
        """ generator of image from skeleton """
        self.netG.eval()
        with torch.no_grad():
            ske_t = self.dataset.preprocessSkeleton(ske)
            ske_t = ske_t.to(self.device)
            ske_t_batch = ske_t.unsqueeze(0)        # make a batch
            normalized_output = self.netG(ske_t_batch)
            normalized_output = torch.Tensor.cpu(normalized_output)
            res = self.dataset.tensor2image(normalized_output[0])       # get image 0 from the batch
        return res




if __name__ == '__main__':
    args = sys.argv
    print(f'args are {args}')
    force = False
    n_epoch = int(args[1])
    train = True
    if len(args) > 2:
        test_opcv = args[2] == '--test'
    else:
        test_opcv = False
    
    if len(args) > 3:
        batch_size = int(args[3])
    else:
        batch_size = 32
    
    if len(sys.argv) > 4:
        filename = sys.argv[4]
        if len(sys.argv) > 5:
            force = sys.argv[5].lower() == "true"
    else:
        filename = "data/taichi1.mp4"
    print("GenVanillaNNImage: Current Working Directory=", os.getcwd())
    print("GenVanillaNNImage: Filename=", filename)
    print("GenVanillaNNImage: Filename=", filename)

    targetVideoSke = VideoSkeleton(filename)

    if train:
        # Train
        gen = GenVanillaNNImage(targetVideoSke, loadFromFile=False, batch_size=batch_size)
        gen.train(n_epoch)
    else:
        gen = GenVanillaNNImage(targetVideoSke, loadFromFile=True)    # load from file        


    # Test with a second video
    if test_opcv:
        for i in range(targetVideoSke.skeCount()):
            if i % 30 == 0:
                image = gen.generate( targetVideoSke.ske[i] )
                #image = image*255
                nouvelle_taille = (256, 256) 
                image = cv2.resize(image, nouvelle_taille)
                cv2.imshow('Image', image)
                key = cv2.waitKey(0)
        cv2.destroyAllWindows()
