"""
Σε αυτή την άσκηση υπάρχει κρυμμένη μια φωτογραφία 128x128 και έχετε σαν στόχο να την εντοπίσετε μέσω ερωτήσεων.
Μπορείτε να δώσετε μια δικιά σας φωτογραφία και θα πάρετε σαν απάντηση ένα similarity score με τη φωτογραφία στόχο.
Κάνοντας πολλές ερωτήσεις μπορείτε να βελτιώσετε την αρχική εκτίμηση ώστε να μοιάζει όλο και περισσότερο στο στόχο.
Υλοποιείστε έναν αλγόριθμο που χρησιμοποιεί τις απαντήσεις της συνάρτησης score για να τρέξει gradient descent (0-order)
στα pixels της φωτογραφίας. Σας δίνεται ένα template κώδικα με 3 διαφορετικές συναρτήσεις ομοιότητας. 
Πόσες φορές χρειάζεται να καλέσετε τη συνάρτηση σε κάθε περίπτωση;
"""
import torch
from PIL import Image
from torchvision import transforms, models

# Load Image and convert it to tensor 3x128x128
SZ = 128
auth = Image.open("mario.png").resize((SZ, SZ)).convert('RGB')  # Convert image to 128x128
transform = transforms.ToTensor()  # convert image to Tensor 3x128x128
transformBack = transforms.ToPILImage()
Y = transform(auth)

# Define the similarity function, different versions according to difficulty 
task = 'simple'  # Change this to 'advanced' or 'hard' as needed

if task == 'simple':
    # Similarity based on l2 distance
    similarity = lambda X, Y: 1 - torch.sqrt(torch.mean((X - Y) * (X - Y)))
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
    if s > cutoff:
        print("Succeeded in", calls, "calls")
        return True
    return False

# Hyperparameters
learning_rate = 0.1
perturbation = 0.01
max_iterations = 1000

# Start with a simple tensor X of all 1s
X = torch.ones((3, 128, 128), requires_grad=False)

for iteration in range(max_iterations):
    current_score = similarity(X, Y)
    print(f"Iteration {iteration}, Current Score: {current_score.item()}")

    if score(X):
        break

    # Create a perturbation tensor
    perturb = torch.zeros_like(X)
    
    # Perturb each pixel to estimate the gradient
    for i in range(3):
        for j in range(128):
            for k in range(128):
                original_value = X[i, j, k].item()
                X[i, j, k] += perturbation
                new_score = similarity(X, Y)
                calls += 1
                perturb[i, j, k] = (new_score - current_score) / perturbation
                X[i, j, k] = original_value  # Reset to original value
    
    # Update X in the direction of the perturbation
    X += learning_rate * perturb
    
    # Ensure X remains within valid image range [0, 1]
    X = torch.clamp(X, 0, 1)

# Save the resulting image
result_image = transformBack(X)
result_image.save("result_image.png")

print("Failed to reach the target similarity within the maximum number of iterations.")
print(f"Total number of similarity function calls: {calls}")
