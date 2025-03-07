import torch
import pickle
import os
from args import get_parser
from model import get_model
from utils.output_utils import prepare_output
from torchvision import transforms
from PIL import Image
from tensorflow.keras.preprocessing import image

# Set up device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
map_loc = None if torch.cuda.is_available() else 'cpu'

# Load vocab files
with open('ingr_vocab.pkl', 'rb') as f:
    ingrs_vocab = pickle.load(f)

with open('instr_vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)

#vocab = pickle.load('instr_vocab.pkl', 'rb')

ingr_vocab_size = len(ingrs_vocab)
instrs_vocab_size = len(vocab)

# Load model
args = get_parser()
args.maxseqlen = 15
args.ingrs_only = False
model = get_model(args, ingr_vocab_size, instrs_vocab_size)
model.load_state_dict(torch.load('modelbest.ckpt', map_location=map_loc))
model.to(device)
model.eval()

# Define image transformations
to_input_transf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224)
])

def process_image(file_path):
    img = image.load_img(file_path)
    image_transf = transform(img)
    image_tensor = to_input_transf(image_transf).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model.sample(image_tensor, greedy=True, temperature=1.0, beam=-1, true_ingrs=None)
    
    ingr_ids = outputs['ingr_ids'].cpu().numpy()
    recipe_ids = outputs['recipe_ids'].cpu().numpy()
    
    outs, valid = prepare_output(recipe_ids[0], ingr_ids[0], ingrs_vocab, vocab)
    
    return outs if valid['is_valid'] else {"title": "Not a valid recipe!", "reason": valid['reason']}

file_path = "/mnt/d/invrec/images/vegetable ramen/vegetable ramen_2.jpg" 
output = process_image(file_path)
print(output["ingrs"])
