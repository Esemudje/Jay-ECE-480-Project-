import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

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
    mfcc_list = []
    
    # Iterate through the required lines in the block
    for i in range(block_start, block_start + block_size):
        frame = lines[i].strip()
        if frame:  # Ignore blank lines
            parts = list(map(float, frame.split()))
            mfcc_list.append(parts)

    return np.array(mfcc_list)

# Function to get starting line for each digit
def get_block_start_for_digit(digit, blocks_per_digit=660):
    start_index = digit * blocks_per_digit
    return sum(block_counts[:start_index])

# Train GMMs for all digits
gmms = {}
for digit in range(10):
    # Combine all utterances for this digit
    digit_data = np.vstack([
        read_mfccs_for_block(
            all_lines,
            get_block_start_for_digit(digit) + sum(block_counts[digit * 660:(digit * 660) + i]),
            block_counts[(digit * 660) + i]
        )
        for i in range(660)
    ])
    
    # Initialize GMM with K-Means
    kmeans = KMeans(n_clusters=2, random_state=0).fit(digit_data)
    gmm = GaussianMixture(
        n_components=2,
        covariance_type='spherical',
        random_state=0,
        means_init=kmeans.cluster_centers_
    )
    gmm.fit(digit_data)
    gmms[digit] = gmm

# Predict digit for each utterance
predictions = []
true_labels = []
for true_digit in range(10):
    for i in range(660):
        block_start = get_block_start_for_digit(true_digit) + sum(block_counts[true_digit * 660:(true_digit * 660) + i])
        block_size = block_counts[(true_digit * 660) + i]
        utterance_data = read_mfccs_for_block(all_lines, block_start, block_size)
        
        # Evaluate log-likelihoods across all GMMs
        log_likelihoods = {digit: gmms[digit].score_samples(utterance_data).sum() for digit in range(10)}
        predicted_digit = max(log_likelihoods, key=log_likelihoods.get)
        
        predictions.append(predicted_digit)
        true_labels.append(true_digit)

# Calculate accuracy
accuracy = np.mean(np.array(predictions) == np.array(true_labels))
print(f"Accuracy: {accuracy:.2%}")

# Visualize confusion matrix
cm = confusion_matrix(true_labels, predictions, labels=range(10))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(10))
disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
plt.title("Spherical Confusion Matrix")
plt.savefig("Spherical Confusion Matrix")
plt.show()
