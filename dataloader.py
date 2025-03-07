import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from nltk.tokenize import word_tokenize
from collections import Counter
import nltk
import json
import os
from PIL import Image
import torchvision.transforms as transforms

nltk.download('punkt')

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class RecipeTextDataset(Dataset):
    # def __init__(self, json_file, image_folder, instr_vocab=None, max_seq_length=20, min_freq=1, img_size=224):
    #     self.json_file = json_file
    #     with open(self.json_file, 'r') as fp:
    #         data = json.load(fp)

    #     self.data = data
    #     self.image_folder = image_folder
    #     self.subfolders = os.listdir(self.image_folder)

    #     self.entries = []  # Store (image_path, directions, ingredients)

    #     for i, entry in enumerate(self.data):
    #         dish_name = entry["name"]
    #         ingredients = entry["ingredients"]
    #         directions = entry["directions"]

    #         # Check if the dish name matches any subfolder
    #         subfolder_path = os.path.join(self.image_folder, dish_name)

    #         if os.path.exists(subfolder_path):  # Ensure subfolder exists
    #             image_files = [f for f in os.listdir(subfolder_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
                
    #             for img in image_files:
    #                 img_path = os.path.join(subfolder_path, img)
    #                 self.entries.append((img_path, directions, ingredients))

    #     self.max_seq_length = max_seq_length
    #     self.img_size = img_size

    #     # Define image transformations
    #     self.transform = transforms.Compose([
    #         transforms.Resize((img_size, img_size)),  # Resize to fixed size
    #         transforms.ToTensor(),  # Convert to Tensor
    #         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize like ImageNet
    #     ])

    #     if instr_vocab is None:
    #         self.instr_vocab = self.build_vocab([e[1] for e in self.entries], min_freq)  # Only using directions for vocab
    #     else:
    #         self.instr_vocab = instr_vocab

    def __init__(self, json_file, image_folder, instr_vocab=None, ingr_vocab=None, max_seq_length=20, min_freq=1, img_size=224):
        self.json_file = json_file
        with open(self.json_file, 'r') as fp:
            data = json.load(fp)

        self.data = data
        self.image_folder = image_folder
        self.subfolders = os.listdir(self.image_folder)

        self.entries = []  # Store (image_path, directions, ingredients)

        for entry in self.data:
            dish_name = entry["name"]
            ingredients = entry["ingredients"]
            directions = entry["directions"]

            subfolder_path = os.path.join(self.image_folder, dish_name)
            if os.path.exists(subfolder_path):
                image_files = [f for f in os.listdir(subfolder_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
                
                for img in image_files:
                    img_path = os.path.join(subfolder_path, img)
                    self.entries.append((img_path, directions, ingredients))

        self.max_seq_length = max_seq_length
        self.img_size = img_size

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        if instr_vocab is None:
            self.instr_vocab = self.build_vocab([e[1] for e in self.entries], min_freq)
        else:
            self.instr_vocab = instr_vocab
        
        if ingr_vocab is None:
            self.ingr_vocab = self.build_vocab([e[2] for e in self.entries], min_freq)
        else:
            self.ingr_vocab = ingr_vocab

    # def build_vocab(self, captions, min_freq):
    #     word_counts = Counter()
    #     for caption in captions:
    #         tokens = word_tokenize(caption.lower())
    #         word_counts.update(tokens)

    #     vocab = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3}  # Special tokens
    #     for word, count in word_counts.items():
    #         if count >= min_freq:
    #             vocab[word] = len(vocab)  # Assign next available index

    #     return vocab

    def build_vocab(self, texts, min_freq):
        word_counts = Counter()
        for text in texts:
            tokens = word_tokenize(text.lower())
            word_counts.update(tokens)

        vocab = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3}
        for word, count in word_counts.items():
            if count >= min_freq:
                vocab[word] = len(vocab)

        return vocab

    def tokenize_and_encode(self, text):
        tokens = word_tokenize(text.lower())  # Tokenize
        indexed_caption = [self.instr_vocab.get("<sos>", 1)] + \
                          [self.instr_vocab.get(token, self.instr_vocab.get("<unk>", 3)) for token in tokens] + \
                          [self.instr_vocab.get("<eos>", 2)]

        indexed_caption = torch.tensor(indexed_caption, dtype=torch.long)

        # Padding/Truncation
        padded_caption = F.pad(indexed_caption, (0, self.max_seq_length - len(indexed_caption)), value=self.instr_vocab.get("<pad>", 0))
        return padded_caption[:self.max_seq_length]

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        img_path, directions, ingredients = self.entries[idx]

        # Load and transform image
        image = Image.open(img_path).convert("RGB")  # Ensure RGB format
        image = self.transform(image)  # Apply transformations

        processed_directions = self.tokenize_and_encode(directions)
        processed_ingredients = self.tokenize_and_encode(ingredients)

        return image, processed_directions, processed_ingredients

# def get_dataloader(json_file, image_folder, batch_size=32, max_seq_length=20, min_freq=1, img_size=224, shuffle=True):
#     dataset = RecipeTextDataset(json_file, image_folder, max_seq_length=max_seq_length, min_freq=min_freq, img_size=img_size)
#     instr_vocab = dataset.instr_vocab  # Get the generated vocabulary
#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
#     return dataloader, instr_vocab

def get_dataloader(json_file, image_folder, batch_size=128, max_seq_length=20, min_freq=1, img_size=224, shuffle=True):
    dataset = RecipeTextDataset(json_file, image_folder, max_seq_length=max_seq_length, min_freq=min_freq, img_size=img_size)
    instr_vocab = dataset.instr_vocab  
    ingr_vocab = dataset.ingr_vocab  

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader, instr_vocab, ingr_vocab

data_loader, instr_vocab , ingr_vocab = get_dataloader("indian_recipes.json" , "/mnt/d/invrec/New/images")

ingr_vocab_size = len(instr_vocab) 
instrs_vocab_size = len(ingr_vocab)  

i = next(iter(data_loader))
img , inst , ingr = i 
from model import get_model 
from args import get_parser 
args = get_parser()
model = get_model(args  , instrs_vocab_size , instrs_vocab_size)

# model.load_state_dict(torch.load("modelbest.ckpt", map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu") ) , strict = False)

checkpoint = torch.load("modelbest.ckpt", map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
model_dict = model.state_dict()

# Filter out layers that don’t match
pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict and model_dict[k].shape == v.shape}

# Update the model’s state_dict
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)

num_epochs = 50 


import torch.optim as optim
from tqdm import tqdm 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define optimizer
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

# Training loop
for epoch in (range(num_epochs)):
    model.train()  # Set model to training mode
    total_loss = 0
    
    for batch_id, (image, processed_directions, processed_ingredients) in enumerate(tqdm(data_loader)):
        # Move data to GPU if available

        image, processed_directions, processed_ingredients = (
            image.to(device),
            processed_directions.to(device),
            processed_ingredients.to(device)
        )
        
        optimizer.zero_grad()  # Reset gradients
        losses = model(image, processed_directions, processed_ingredients)
        
        # Extract individual losses
        ingr_loss = losses["ingr_loss"].mean()
        recipe_loss = losses["recipe_loss"].mean()
        card_penalty = losses["card_penalty"].mean()
        eos_loss = losses["eos_loss"].mean()
        iou_loss = losses["iou"].mean()
        
        # Compute total loss
        total_batch_loss = ingr_loss + recipe_loss + card_penalty + eos_loss + iou_loss
        total_batch_loss.backward()  # Backpropagate
        optimizer.step() 
        
        total_loss += total_batch_loss.item()
        
        # if batch_id % 10 == 0:  # Print progress every 10 batches
        #     print(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_id}/{len(data_loader)}], Loss: {total_batch_loss.item():.4f}")
    
    avg_loss = total_loss / len(data_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}] Completed. Average Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), f"model_{epoch + 11}.pth")
               

