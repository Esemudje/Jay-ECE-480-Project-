import numpy as np
import matplotlib.pyplot as plt

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

# ------------------------------------------------------------------------------------

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

# ------------------------------------------------------------------------------------

# Function to get the block start index for a given digit
def get_block_start_for_digit(digit, blocks_per_digit=660):
    # Each digit corresponds to 660 blocks, so we can compute the starting block
    return sum(block_counts[:digit * blocks_per_digit])

# ------------------------------------------------------------------------------------

# Create subplots for each digit
fig, axes = plt.subplots(2, 5, figsize=(20, 8))  # 2 rows, 5 columns
axes = axes.flatten()  # Flatten axes array for easier indexing

for digit in range(10):
    with open(filename, "r") as file:
        block_start = get_block_start_for_digit(digit)  # Get the starting index for the digit
        mfccOneList, mfccTwoList, mfccThreeList = read_mfccs_for_block(file, block_start, block_counts[digit * 660])

        analysisWindow = np.linspace(0, len(mfccOneList), len(mfccOneList))
        
        # Plot MFCCs for the digit in the corresponding subplot
        axes[digit].plot(analysisWindow, mfccOneList, label='MFCC 1')
        axes[digit].plot(analysisWindow, mfccTwoList, label='MFCC 2')
        axes[digit].plot(analysisWindow, mfccThreeList, label='MFCC 3')
        axes[digit].set_title(f"Digit {digit}")
        axes[digit].set_xlabel("Analysis Window")
        axes[digit].set_ylabel("MFCC Values")
        axes[digit].legend(loc='upper right')

# Adjust layout and save the figure
plt.tight_layout()
plt.savefig("mfccs_digits_5x2.png")
plt.show()
