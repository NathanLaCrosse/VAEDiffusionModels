import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import NetworkComponents as nc
import mushroomdata
from unet import UNET
from NathanVAE import VAE
import matplotlib.pyplot as plt

cpu = torch.device('cpu')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vae = VAE(8)
unet = UNET(64, 128, 100)

vae.load_state_dict(torch.load("PTFiles/largernorm3.pt", map_location=device))
unet.load_state_dict(torch.load("PTFiles/inprogress49twohundred.pt", map_location=device))
vae = vae.to(device)
unet = unet.to(device)

start_step = 0.0001
end_step = 0.02
num_time_steps = 250

betas = np.linspace(start_step, end_step, num_time_steps)
alphas = 1 - betas
alpha_bars = np.zeros(num_time_steps)
alpha_bars[0] = alphas[0]
for i in range(1, num_time_steps):
    alpha_bars[i] = alphas[i] * alpha_bars[i-1]
time_encodings = nc.positional_encoding(num_time_steps, 64)

#Graph components
rows = 2
cols = 2


# Method to decode latent -> formula from class
def denoise_latent(latent, unet, alphas, betas, alpha_bars, time_encodings, total_noise_steps, label):
    bs, _, _, _ = latent.size()
    pred = latent
    t = total_noise_steps

    with torch.no_grad():
        while t > 0:
            step_vect = time_encodings[t-1].unsqueeze(0).expand(bs, 64).to(device)
            noise = unet(pred, step_vect, label)

            pred = 1 / np.sqrt(alphas[t-1]) * (pred - betas[t-1] / np.sqrt(1 - alpha_bars[t-1]) * noise)

            if t > 1:
                pred = pred + np.sqrt(betas[t-1]) * torch.randn_like(pred)            

            t -= 1
        
        return pred


def denoise_step_by_step(latent, unet, alphas, betas, alpha_bars, time_encodings, total_noise_steps, label):
    bs, _, _, _ = latent.size()
    pred = latent
    t = total_noise_steps
    fig, ax = plt.subplots(rows, cols)
    title = fig.suptitle(f"Denoising Step: {total_noise_steps}", fontsize=16)
    with torch.no_grad():
       original_noise  = vae.forward_decode_only(pred)

    # makes it so that the plot is drawing from reference instead if instantiating a new image every time
    image_references = []

    #plot the original noise, and set intialization for image location references
    for i in range(rows):
        for j in range(cols):
            picture = original_noise[i * cols + j].to(cpu)
            picture = (picture + 1) / 2
            picture = picture.permute(1, 2, 0)#.clamp(0, 1).numpy()
            ref = ax[i, j].imshow(picture)
            ax[i, j].axis("off")
            image_references.append(ref)

    plt.tight_layout()
    plt.pause(0.001)

    with torch.no_grad():
        while t > 0:
            title.set_text(f"Denoising Step: {t}")
            step_vect = time_encodings[t - 1].unsqueeze(0).expand(bs, 64).to(device)
            noise = unet(pred, step_vect, label)
            pred = 1 / np.sqrt(alphas[t - 1]) * (
                    pred - betas[t - 1] / np.sqrt(1 - alpha_bars[t - 1]) * noise
            )
            if t > 1:
                pred = pred + np.sqrt(betas[t - 1]) * torch.randn_like(pred)

            # decode then plot the decoding step
            decoded_pictures = vae.forward_decode_only(pred)

            for i in range(rows):
                for j in range(cols):
                    picture = decoded_pictures[i * cols + j].to(cpu)
                    picture = (picture + 1) / 2
                    picture = picture.permute(1, 2, 0) #.clamp(0, 1).numpy()
                    image_references[i * cols + j].set_data(picture)

            #update the canvas without remaking the window, with a pause
            fig.canvas.draw_idle()
            plt.pause(0.1)

            #go down one time step
            t  = t-1
    return pred

def plot_final_result():
# Actual testing stuff here
    with torch.no_grad():
        while True:
            rows = 2
            cols = 2

            samp = torch.randn((rows*cols, 8, 8, 8), device=device)
            labels = torch.randint(0,100,(rows*cols,), device=device)

            denoised = denoise_latent(samp, unet, alphas, betas, alpha_bars, time_encodings, num_time_steps, labels)
            # denoised = F.normalize(denoised, dim=-1)
            # print(denoised.mean().item(), denoised.std().item())
            ims = vae.forward_decode_only(denoised)

            fig, ax = plt.subplots(rows, cols)

            for i in range(rows):
                for k in range(cols):
                    im = ims[i*rows + k].to(cpu)
                    im = (im + 1) / 2
                    im = im.permute(1,2,0)

                    ax[i, k].imshow(im)
                    ax[i, k].axis('off')

            plt.tight_layout()
            plt.show()

def plot_denoising_animation():
    with torch.no_grad():
        bs = rows * cols
        samp = torch.randn((bs, 8, 8, 8), device=device)
        labels = torch.randint(0, 100, (bs,), device=device)

        #latent, unet, alphas, betas, alpha_bars, time_encodings, total_noise_steps, label
        denoise_step_by_step(samp, unet, alphas, betas, alpha_bars, time_encodings, num_time_steps, labels)


plot_denoising_animation()