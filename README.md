# Attempt at Stable Diffusion
Authors: Nathan LaCrosse, Matthew Peplinski, and Jake Swanson

## Our Project:
We attempt to recreate the stable diffusion model from this [paper](https://arxiv.org/abs/2006.11239) by Berkeley University.

We utilize a dataset from [Kaggle](https://www.kaggle.com/datasets/thehir0/mushroom-species) that contains images of mushrooms. There are various different species of mushrooms. This is perfect for being able to utilize the hybridization capabilities of a diffusion model.

## Architecture:
The model runs off of two different base models. A UNet and VAE architecture. The high level overview is the VAE is capable of generating latent vectors of the original images. These latent's are fed into the UNet on a denoising schedule to essentially find a mushroom in random noise.

### UNet:

### VAE:

## Results:
Below is a sample of the results we got from our model.

<p>Sample of 64 x 64 results:</p>
![64SizedImages](https://github.com/NathanLaCrosse/VAEDiffusionModels/blob/main/Mushroom64by3x3.png)

<p>Samples of 128 x 128 results:</p>
![128SizedImages1](https://github.com/NathanLaCrosse/VAEDiffusionModels/blob/main/Mushroom128by3x3.png)

![128SizedImages2](https://github.com/NathanLaCrosse/VAEDiffusionModels/blob/main/Mushroom128by2x2.png)
