import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

filename = "Train_Arabic_digit.txt"

# Function to count lines per block
def count_lines_per_block(file):
    block_counts = []
    current_block_count = 0
    
    for line in file:
        if line.strip():  # If not a blank line
            current_block_count += 1
        else:
            if current_block_count > 0:
                block_counts.append(current_block_count)
                current_block_count = 0  # Reset for next block

    # Add the last block if it didn't end with a blank line
    if current_block_count > 0:
        block_counts.append(current_block_count)
    
    return block_counts

# Read block counts from file
with open(filename, "r") as file:
    block_counts = count_lines_per_block(file)

# Function to read MFCCs for a specific block
def read_mfccs_for_block(file, block_start, block_size):
    mfccOneList = []
    mfccTwoList = []
    mfccThreeList = []

    # Skip lines to reach the desired block
    for _ in range(block_start):
        file.readline()

    file.readline()  # Skip blank line that separates blocks

    # Read lines within the block
    for _ in range(block_size):
        frame = file.readline()
        if frame.strip():  # If not a blank line
            mfccOne = frame.split(" ")[0]  # First MFCC
            mfccTwo = frame.split(" ")[1]  # Second MFCC
            mfccThree = frame.split(" ")[2]  # Third MFCC

            mfccOneList.append(float(mfccOne))
            mfccTwoList.append(float(mfccTwo))
            mfccThreeList.append(float(mfccThree))

    return mfccOneList, mfccTwoList, mfccThreeList

# Function to move to the correct block for a given digit
def get_block_start_for_digit(digit, blocks_per_digit=660):
    # Each digit corresponds to 660 blocks, so we can compute the starting block
    return sum(block_counts[:digit * blocks_per_digit])

# Plot for the nth block (digit n)
digit = 0
with open(filename, "r") as file:
    mfccOneList, mfccTwoList, mfccThreeList = read_mfccs_for_block(file, digit, block_counts[0])
    
    # Combine the features into a single array for clustering
    data = np.array([mfccOneList, mfccTwoList, mfccThreeList]).T

    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=4, random_state=42)
    labels = kmeans.fit_predict(data)

    # work on sizing
    fig, axs = plt.subplots(1, 3, figsize=(10, 3))
    fig.suptitle(f'Comparison of MFCCs for Digit {digit} with K-Means', fontsize=16)

    # Scatter plots with color encoding by cluster labels
    axs[0].scatter(mfccOneList, mfccTwoList, c=labels, cmap='viridis', s=10)
    axs[0].set_title('MFCC1 vs MFCC2')
    axs[0].set_xlabel('MFCC1')
    axs[0].set_ylabel('MFCC2')

    axs[1].scatter(mfccOneList, mfccThreeList, c=labels, cmap='viridis', s=10)
    axs[1].set_title('MFCC1 vs MFCC3')
    axs[1].set_xlabel('MFCC1')
    axs[1].set_ylabel('MFCC3')

    axs[2].scatter(mfccTwoList, mfccThreeList, c=labels, cmap='viridis', s=10)
    axs[2].set_title('MFCC2 vs MFCC3')
    axs[2].set_xlabel('MFCC2')
    axs[2].set_ylabel('MFCC3')

    plt.tight_layout()
    plt.savefig("0_MFCC_Kmeans")
    plt.show()
