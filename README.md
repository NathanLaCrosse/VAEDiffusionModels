# Diffushroom: A Stable Diffusion model for Mushrooms
Authors: Nathan LaCrosse, Matthew Peplinski, and Jake Swanson

## Our Project:
We attempt to recreate the stable diffusion model from this [paper](https://arxiv.org/abs/2006.11239) by Berkeley University.

We utilize a dataset from [Kaggle](https://www.kaggle.com/datasets/thehir0/mushroom-species) that contains images of mushrooms. There are various different species of mushrooms. This is perfect for being able to utilize the hybridization capabilities of a diffusion model.

## Architecture:
The model runs off of two different base models. A UNet and VAE architecture. The high level overview is the VAE is capable of generating latent vectors of the original images. These latent's are fed into the UNet on a denoising schedule to essentially find a mushroom in random noise

### VAE:
The goal of the VAE is to encode the image information into a smaller latent space which is normally distributed by using KL divergence and be able to reconstruct the encoded image with minimal information loss. Our VAE used residual blocks to compress latents down to 4x16x16 from 3-color channel 64x64 pixel images, and 4x32x32 sized latents from the 128x128 3-color channel images to match similar latent dimentions to the stable diffusion paper. 

### UNET:
The goal of the unet is to improve samples from the unet. These samples look very noisy, so we utilize the UNet architecutre as a denoiser. In this architecture, an image is iteratively analyzed at smaller and smaller scales to gather global features that is concatenated with local details. However, to be a proper diffusion model, this architecture has been modified to accept a time embedding and a label embedding. Starting with the time embedding, the unet denoises using a noise scheduler described in the previously mentioned paper. Each time step is passed through a modified transformer positional encoding and passed through a multi-layer perceptron to allow the model to learn a custom time embedding. Each label is passed through an embedding layer and another multi-layer perceptron.

In each UNet block, we incorporate the following:
  1. Two residual layers that apply a 1x1 convolution, a 3x3 convolution and a final 1x1 convolution. Before the 3x3 convolution, a linearly transformed version of the time embedding is added to the image. SiLU is used as an activation function and uses groupnorm for normalization.
  2. A cross attention mechanism, which is a form of transformer-style attention where a linearly transformed label embedding creates the keys and values for the attention mechanism and the pixel values (across all channels) form the queries. In other words, each pixel is allowed to "talk" to the label vector.
  3. A self attention mechanism, another form of transformer-style attention in which each pixel (across all channels), is used to generate queries, keys and values. In other words, each pixel "talks" to each other pixel.

## Results:
Below is a sample of the results we got from our model.

<p>Sample of 64 x 64 results:</p>

![64SizedImages](https://github.com/NathanLaCrosse/VAEDiffusionModels/blob/main/Mushroom64by3x3.png)

<p>Samples of 128 x 128 results:</p>

![128SizedImages1](https://github.com/NathanLaCrosse/VAEDiffusionModels/blob/main/Mushroom128by3x3.png)

![128SizedImages2](https://github.com/NathanLaCrosse/VAEDiffusionModels/blob/main/Mushroom128by2x2.png)
