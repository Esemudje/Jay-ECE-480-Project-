import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

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

    if current_block_count > 0:
        block_counts.append(current_block_count)
    
    return block_counts

# Read block counts from file
with open(filename, "r") as file:
    block_counts = count_lines_per_block(file)

# Function to read MFCCs for a specific block
def read_mfccs_for_block(file, block_start, block_size):
    mfccOneList, mfccTwoList, mfccThreeList = [], [], []

    for _ in range(block_start):
        file.readline()
    file.readline()  # Skip blank line that separates blocks

    for _ in range(block_size):
        frame = file.readline()
        if frame.strip():
            mfccOne, mfccTwo, mfccThree = frame.split()[:3]
            mfccOneList.append(float(mfccOne))
            mfccTwoList.append(float(mfccTwo))
            mfccThreeList.append(float(mfccThree))

    return np.array(mfccOneList), np.array(mfccTwoList), np.array(mfccThreeList)

# Function to get block start for each digit
def get_block_start_for_digit(digit, blocks_per_digit=660):
    return sum(block_counts[:digit * blocks_per_digit])

# Loop through each digit (0 to 9)
for digit in range(10):
    with open(filename, "r") as file:
        block_start = get_block_start_for_digit(digit)
        mfcc1, mfcc2, mfcc3 = read_mfccs_for_block(file, block_start, block_counts[digit * 660])

        # Stack MFCCs into a single array for clustering
        data = np.column_stack((mfcc1, mfcc2, mfcc3))

        # Initialize GMM using K-Means with 2 clusters
        kmeans = KMeans(n_clusters=4, random_state=50).fit(data)
        gmm = GaussianMixture(n_components=4, covariance_type='full', random_state=10, means_init=kmeans.cluster_centers_)
        gmm.fit(data)

        # Set up 3 subplots for the 3 pairs of MFCCs
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f'K-Means Contours for Digit {digit}', fontsize=16)

        # Define the grid for contour plots
        x = np.linspace(min(mfcc1), max(mfcc1), 100)
        y = np.linspace(min(mfcc2), max(mfcc2), 100)
        X, Y = np.meshgrid(x, y)

        # MFCC1 vs MFCC2
        XX = np.column_stack((X.ravel(), Y.ravel(), np.mean(mfcc3) * np.ones(X.ravel().shape)))
        Z = np.exp(gmm.score_samples(XX)).reshape(X.shape)
        axs[0].contourf(X, Y, Z, levels=10, cmap='viridis')
        axs[0].scatter(mfcc1, mfcc2, c='blue', s=5, alpha=0.5)
        axs[0].set_title('MFCC1 vs MFCC2')
        axs[0].set_xlabel('MFCC1')
        axs[0].set_ylabel('MFCC2')

        # MFCC1 vs MFCC3
        x = np.linspace(min(mfcc1), max(mfcc1), 100)
        y = np.linspace(min(mfcc3), max(mfcc3), 100)
        X, Y = np.meshgrid(x, y)
        XX = np.column_stack((X.ravel(), np.mean(mfcc2) * np.ones(X.ravel().shape), Y.ravel()))
        Z = np.exp(gmm.score_samples(XX)).reshape(X.shape)
        axs[1].contourf(X, Y, Z, levels=10, cmap='viridis')
        axs[1].scatter(mfcc1, mfcc3, c='green', s=5, alpha=0.5)
        axs[1].set_title('MFCC1 vs MFCC3')
        axs[1].set_xlabel('MFCC1')
        axs[1].set_ylabel('MFCC3')

        # MFCC2 vs MFCC3
        x = np.linspace(min(mfcc2), max(mfcc2), 100)
        y = np.linspace(min(mfcc3), max(mfcc3), 100)
        X, Y = np.meshgrid(x, y)
        XX = np.column_stack((np.mean(mfcc1) * np.ones(X.ravel().shape), X.ravel(), Y.ravel()))
        Z = np.exp(gmm.score_samples(XX)).reshape(X.shape)
        axs[2].contourf(X, Y, Z, levels=10, cmap='viridis')
        axs[2].scatter(mfcc2, mfcc3, c='red', s=5, alpha=0.5)
        axs[2].set_title('MFCC2 vs MFCC3')
        axs[2].set_xlabel('MFCC2')
        axs[2].set_ylabel('MFCC3')

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        #plt.savefig(f"kmeans_distinct_spherical_{digit}.png")
        plt.show()
