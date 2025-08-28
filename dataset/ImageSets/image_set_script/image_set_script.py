import random

# Step 1: Generate all numbers from 1 to 449
numbers = list(range(1, 450))  # [1, 2, 3, ..., 449]

# Shuffle to randomize selection
random.shuffle(numbers)

# Define 80% split
split_80 = int(0.8 * len(numbers))  # 0.8 * 549 = 439.2 → 439

file1_numbers = sorted(numbers[:split_80])    # First 80% - sorted
file2_numbers = sorted(numbers[split_80:])    # Remaining 20% - sorted
all_numbers_sorted = sorted(numbers)          # Same as range(1,550), but for consistency

# Write File 1 (80%)
with open("file1_80percent.txt", "w") as f1:
    f1.write("\n".join(map(str, file1_numbers)))

# Write File 2 (20%)
with open("file2_20percent.txt", "w") as f2:
    f2.write("\n".join(map(str, file2_numbers)))

# Write File 3: All numbers 1 to 549, sorted
with open("all_numbers_sorted.txt", "w") as f3:
    f3.write("\n".join(map(str, all_numbers_sorted)))

# Confirmation
print(f"File 1 (80%): {len(file1_numbers)} numbers → file1_80percent.txt")
print(f"File 2 (20%): {len(file2_numbers)} numbers → file2_20percent.txt")
print(f"All numbers: {len(all_numbers_sorted)} numbers → all_numbers_sorted.txt")
print("All files created successfully.")