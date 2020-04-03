import torch
import torchvision
from PIL import Image
import os


class VideoSuperResolution(torch.utils.data.Dataset):
    def __init__(self,
                 resolution_a,
                 resolution_b,
                 root_dir="/data",
                 transform=None,
                 start=46987,
                 stop=47018,
                 eps=1e-2):
        """
        Args:
            resolution (int): Path to the csv file with annotations.
            root_dir   (string): Directory with all the images.
            transform  (callable, optional): Optional transform to be applied
                on a sample.
            start      (int): Starting number in file naming convention.
            stop       (int): Ending number in file naming convention.
            eps        (int): Threshold for minimum pixel-wise difference.
        """
        self.path_a = os.path.join(root_dir, str(resolution_a))
        self.path_b = os.path.join(root_dir, str(resolution_b))
        self.start = start
        self.stop = stop
        self.eps = eps

        self.non_identical_images = self.identify_unique_images(start, stop, self.path_a, eps)

        self.len = len(self.non_identical_images)

        if transform == None:
            self.transform = torchvision.transforms.ToTensor()
        else:
            self.transform = transform

    def __len__(self):
        return self.len // 3

    def __getitem__(self, index):
        i = index * 3

        index = self.non_identical_images[i]

        # Get path with image index
        image_a_path = "{}/frame_{:06d}.png".format(self.path_a, index)
        image_b_path = "{}/frame_{:06d}.png".format(self.path_b, index)

        # Open image at path
        image_a = Image.open(image_a_path).convert('RGB')
        image_b = Image.open(image_b_path).convert('RGB')

        image_a, image_b = self.resize_to_64(image_a, image_b)

        # Transform image
        image_a = self.transform(image_a)
        image_b = self.transform(image_b)

        # Set adjacent first and third frame as input
        X = image_a

        # Set intermediate second frame as output
        Y = image_b

        return X, Y

    def resize_to_64(self, image_a, image_b):
        resized_a = image_a.resize((64, 64))
        resized_b = image_b.resize((64, 64))
        return resized_a, resized_b

    def identify_unique_images(self, start, stop, path, eps):
        non_identical_images = []

        if eps == None:
            return list(range(start, stop + 1))
        else:
            i = start
            while i < stop:
                # Get path with image index
                image_a_path = "{}/frame_{:06d}.png".format(path, i)
                image_b_path = "{}/frame_{:06d}.png".format(path, i + 1)

                # Open image at path
                image_a = Image.open(image_a_path).convert('RGB')
                image_b = Image.open(image_b_path).convert('RGB')

                # Transform image
                image_a = torchvision.transforms.ToTensor()(image_a)
                image_b = torchvision.transforms.ToTensor()(image_b)

                # Base case, j=0, means 0 of the subsequent images are the same
                j = 0

                # Loop continues if image_a and image_b are about the same
                while torch.norm(image_a - image_b) < eps:
                    # increment number of similar subsequent images
                    j = j + 1

                    # Get path with next image index for comparison
                    image_b_path = "{}/frame_{:06d}.png".format(path, i + j + 1)

                    # Open and trasnform image at path for comparison
                    image_b = Image.open(image_b_path).convert('RGB')
                    image_b = torchvision.transforms.ToTensor()(image_b)

                # Append first image in set of corresponding similar images
                non_identical_images.append(i)

                # Increment to start next set of images in next loop iteration
                i = i + j + 1

            return non_identical_images
