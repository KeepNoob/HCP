import os, argparse, torch, datetime

def parse_args():
    """Cofiguration of the arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-dir', type=str, help='Path to the image data', default= r'image')
    parser.add_argument('--logs-dir', type=str, help='Path to store logs', default=r"logs")
    parser.add_argument('--model-dir', type=str, help='Path to store models', default=r"models")
    parser.add_argument('--output-dir', type=str, help='Path to the output', default=r"output")
    parser.add_argument("--pretrained", type=str, help='Path to the pretrained model', default=r"stabilityai/sd-vae-ft-mse")
    parser.add_argument('--image-size', type=int, default=256)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--save_interval', type=int, default=25,)
    parser.add_argument('--hcp_type', type=str, default='PRHCP', help='Type of HCP loss to use')
    parser.add_argument('--num-workers', type=int, default=4, metavar='N',
                        help='number of dataloader workers if device is CPU (default: 8)')
    

  
    args = parser.parse_args()
    args.timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    args.output_dir = os.path.join(args.output_dir, args.timestamp)
    os.makedirs(args.output_dir, exist_ok=True)
    args.image_dir = os.path.join(args.output_dir, 'images')
    args.logs_dir = os.path.join(args.output_dir, 'logs')
    args.model_dir = os.path.join(args.output_dir, 'models')
    os.makedirs(args.image_dir, exist_ok=True)
    os.makedirs(args.logs_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    return args

if __name__ == '__main__':
    args = parse_args()
    print(args)