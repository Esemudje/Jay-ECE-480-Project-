import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

filename = "Train_Arabic_digit.txt"

# Function to count lines per block
def count_lines_per_block(lines):
    block_counts = []
    current_block_count = 0
    
    for line in lines:
        if line.strip():  # If not a blank line
            current_block_count += 1
        else:
            if current_block_count > 0:
                block_counts.append(current_block_count)
                current_block_count = 0  # Reset for next block

    if current_block_count > 0:
        block_counts.append(current_block_count)
    
    return block_counts

# Read entire file at once to avoid misalignment issues
with open(filename, "r") as file:
    all_lines = file.readlines()

# Count lines per block for indexing
block_counts = count_lines_per_block(all_lines)

# Function to read MFCCs for a specific block
def read_mfccs_for_block(lines, block_start, block_size):
    mfccs = []

    # Iterate through the required lines in the block
    for i in range(block_start, block_start + block_size):
        frame = lines[i].strip()
        if frame:  # Ignore blank lines
            parts = frame.split()
            if len(parts) >= 13:  # Ensure there are at least 13 elements
                mfcc_values = list(map(float, parts[:13]))
                mfccs.append(mfcc_values)

    return np.array(mfccs)

# Function to get starting line for each digit
def get_block_start_for_digit(digit, blocks_per_digit=660):
    start_index = digit * blocks_per_digit
    return sum(block_counts[:start_index])

# Train GMM on digit 
digit = 0
block_start = get_block_start_for_digit(digit)
data = read_mfccs_for_block(all_lines, block_start, block_counts[digit * 660])

# Initialize GMM for digit 3 using K-Means with 2 clusters
kmeans = KMeans(n_clusters=2, random_state=0).fit(data)
gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=0, means_init=kmeans.cluster_centers_)
gmm.fit(data)

# List to store log-likelihoods for each digit's utterances
log_likelihoods = {d: [] for d in range(10)}

# Calculate log-likelihoods for each digit's utterances under the GMM for digit 3
for d in range(10):
    for i in range(660):  # Assuming 660 utterances per digit
        block_start = get_block_start_for_digit(d) + sum(block_counts[d * 660:(d * 660) + i])
        block_size = block_counts[(d * 660) + i]
        utterance_data = read_mfccs_for_block(all_lines, block_start, block_size)
        
        # Calculate log-likelihood for the entire utterance by summing log-likelihoods of each frame
        utterance_log_likelihood = gmm.score_samples(utterance_data).sum()
        log_likelihoods[d].append(utterance_log_likelihood)

bin_edges = np.linspace(min(log_likelihoods[d]), max(log_likelihoods[d]), 31)

# Define the desired y-axis bounds
y_min, y_max = 0, 0.00014  # Adjust these values based on your data

# Visualize PDF of log-likelihoods for each digit
fig, axs = plt.subplots(2, 5, figsize=(20, 8))
axs = axs.ravel()
for d in range(10):
    axs[d].hist(log_likelihoods[d], bins=bin_edges, density=True, alpha=0.6, color='b')
    axs[d].set_title(f'Log-Likelihood PDF for Digit {d}')
    axs[d].set_xlabel('Log-Likelihood')
    axs[d].set_ylabel('Density')
    
    # Set the y-axis bounds
    axs[d].set_ylim(y_min, y_max)

plt.suptitle(f"GMM Trained on Digit {digit}")
plt.tight_layout()
plt.savefig(f"Digit_{digit}_GMM.png")
plt.show()
