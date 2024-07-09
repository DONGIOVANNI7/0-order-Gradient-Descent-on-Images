import torch
from PIL import Image
from torchvision import transforms, models
import os
import math

# Load Image and convert it to tensor 3x128x128
SZ = 128
auth = Image.open("mario.png").resize((SZ, SZ)).convert('RGB') # Convert image to 128x128
transform = transforms.ToTensor() # convert image to Tensor 3x128x128
transformBack = transforms.ToPILImage()
Y = transform(auth)

# Define the similarity function, different versions according to difficulty 
task = 'hard'
if task == 'simple':
    # Similarity based on l2 distance
    similarity = lambda X, Y: 1 - torch.sqrt(torch.mean((X - Y) ** 2))
elif task == 'advanced':
    # Cosine Similarity
    similarity = lambda X, Y: torch.cosine_similarity(X.view(-1), Y.view(-1), dim=0)
elif task == 'hard':
    # Cosine Similarity of Embeddings
    net = models.shufflenet_v2_x1_0(weights=models.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1)
    net.fc = torch.nn.Identity()
    similarity = lambda X, Y: torch.cosine_similarity(net(X.unsqueeze(0))[0], net(Y.unsqueeze(0))[0], dim=0).detach()

# Define the score function
# It checks the similarity of the given tensor to the target and succeeds if it is greater than a cutoff
calls = 0
def score(X, cutoff=0.95):
    global calls
    calls += 1
    s = similarity(X, Y)
    return s

# Hyperparameters
max_calls = 25000  # maximum number of function calls
initial_noise_level = 0.1  # initial level of noise to add in each iteration
cooling_rate = 0.999  # slower rate at which to reduce the noise level
num_candidates = 20  # number of candidates to generate in each iteration
temperature = 1.0  # initial temperature for simulated annealing

# Start with a simple tensor Χ of all 1s
X = torch.ones((3, SZ, SZ))

# Iteratively change X to improve the score
best_score = score(X)
best_X = X.clone()

for i in range(max_calls):
    # Reduce the noise level as we progress
    noise_level = initial_noise_level * (cooling_rate ** i)
    
    # Create multiple candidates by adding noise
    candidates = [X + noise_level * torch.randn_like(X) for _ in range(num_candidates)]
    candidates = [torch.clamp(candidate, 0, 1) for candidate in candidates]
    
    # Calculate the similarity scores of all candidates
    candidate_scores = [score(candidate) for candidate in candidates]
    
    # Select the best candidate
    max_score, max_idx = max((val, idx) for (idx, val) in enumerate(candidate_scores))
    
    # If the best candidate is better, update X
    if max_score > best_score or torch.rand(1).item() < math.exp((max_score - best_score) / temperature):
        best_score = max_score
        X = candidates[max_idx]
        best_X = X.clone()
    
    # Decrease the temperature
    temperature *= cooling_rate
    
    # Check if the similarity is greater than the cutoff
    if best_score > 0.95:
        break

# Save the final image
final_image = transformBack(best_X)
final_image.save("final_image.png")
print(f"Succeeded in {calls} calls, final similarity score: {best_score:.4f}")
print("Final image saved as final_image.png")


