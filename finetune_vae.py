from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
import torch
import torchvision
from torchvision.transforms import v2 
from parse_args import parse_args
from tqdm import tqdm 
import torch.nn.functional as F
from loss import rec_loss_function, hcp_loss_fn
from torch.utils.data import DataLoader
import src.utility 
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def train_one_epoch(vae:AutoencoderKL, dataloader, optimizer, device, hcp_type='PRHCP'):
    vae.train()
    total_loss = 0
    for images, _ in dataloader:
        images = images.to(device)
        x = images
        optimizer.zero_grad()
        posterior = vae.encoder(images).latent_dist
        # l = 0.18215 * posterior.sample()
        l =  posterior.sample()
        rec_x = vae.decoder(l) 
        rec_loss = rec_loss_function(rec_x, x, 'MSE') 
        l = l.view(l.size(0), -1)
        hcp_loss = hcp_loss_fn(l, hcp_type)
        loss = rec_loss + hcp_loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss

if __name__ == '__main__':
    args = parse_args()
    set_seed(args.seed)

    # Load the pretrained model
    vae= AutoencoderKL.from_pretrained(args.pretrained)
    vae = vae.to(args.device)
    vae.train()
    # Load the dataset
    transforms = v2.Compose([v2.ToImage(),
                            v2.Resize((args.image_size, args.image_size)),
                            v2.RandomHorizontalFlip(p=0.5),
                            v2.ToDtype(torch.float32, scale=True),
                            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                            ])
    dataset = torchvision.datasets.CIFAR10(root="data", train=True, transform=transforms, download=True)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # optimizer
    optimizer = torch.optim.Adam(vae.parameters(), lr=args.learning_rate) 
    # Fine-tune the model
    for epoch in tqdm(range(args.epochs)):
        total_loss = train_one_epoch(vae, dataloader, vae.optimizer, args.device, hcp_type= args.hcp_type)
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {total_loss:.4f}")
        # Save the model
        if (epoch+1) % args.save_interval == 0:
            torch.save(vae.state_dict(), f"{args.save_dir}/vae_{args.hcp_type}_{epoch+1}.pt")

            
