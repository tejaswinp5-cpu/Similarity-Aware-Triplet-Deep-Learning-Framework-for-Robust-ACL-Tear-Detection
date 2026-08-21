# ================================================================
# ACL KNEE MRI - TRIPLET GENERATION AND SIMILARITY COMPARISON
# ================================================================
#
# Dataset:
#   Train
#   Validation
#   Test
#
# Operations:
#   1. Load MRI images
#   2. Extract ResNet-50 embeddings
#   3. Generate Anchor-Positive-Negative triplets
#   4. Calculate Euclidean distance
#   5. Calculate cosine similarity
#   6. Generate 3, 5 and 15 triplets
#   7. Plot Triplet Distance Measurement
#   8. Save results to CSV
#
# CPU compatible
# ================================================================

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize


# ================================================================
# 1. CONFIGURATION
# ================================================================

TRAIN_DIR = r"C:\Users\Dsu\Desktop\VNA_Research_Work\ACL\Raw Datset Knee MRI slices\images\train"

VAL_DIR = r"C:\Users\Dsu\Desktop\VNA_Research_Work\ACL\Raw Datset Knee MRI slices\images\validation"

TEST_DIR = r"C:\Users\Dsu\Desktop\VNA_Research_Work\ACL\Raw Datset Knee MRI slices\images\test"


# Output directory
OUTPUT_DIR = r"C:\Users\Dsu\Desktop\VNA_Research_Work\ACL\Triplet_Results"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# CPU
DEVICE = torch.device("cpu")

print("Using device:", DEVICE)


# ================================================================
# 2. RANDOM SEED
# ================================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ================================================================
# 3. IMAGE TRANSFORMATION
# ================================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ================================================================
# 4. LOAD IMAGE PATHS
# ================================================================

def collect_images(dataset_path):

    image_extensions = (
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tif",
        ".tiff"
    )

    image_paths = []
    labels = []

    if not os.path.exists(dataset_path):
        print("Directory does not exist:", dataset_path)
        return image_paths, labels

    # Each immediate subfolder represents one class
    class_names = sorted([
        d for d in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, d))
    ])

    print("\nDataset:", dataset_path)
    print("Classes:", class_names)

    for class_name in class_names:

        class_dir = os.path.join(dataset_path, class_name)

        for root, dirs, files in os.walk(class_dir):

            for file in files:

                if file.lower().endswith(image_extensions):

                    image_paths.append(
                        os.path.join(root, file)
                    )

                    labels.append(class_name)

    return image_paths, labels


# ================================================================
# 5. LOAD TRAIN / VALIDATION / TEST
# ================================================================

train_paths, train_labels = collect_images(TRAIN_DIR)

val_paths, val_labels = collect_images(VAL_DIR)

test_paths, test_labels = collect_images(TEST_DIR)


print("\n==============================================")
print("DATASET SUMMARY")
print("==============================================")

print("Training images   :", len(train_paths))
print("Validation images :", len(val_paths))
print("Testing images    :", len(test_paths))


# ================================================================
# 6. DISPLAY CLASS DISTRIBUTION
# ================================================================

def class_distribution(labels, dataset_name):

    if len(labels) == 0:
        return

    df = pd.Series(labels).value_counts()

    print("\n", dataset_name)
    print(df)


class_distribution(train_labels, "Training distribution")
class_distribution(val_labels, "Validation distribution")
class_distribution(test_labels, "Testing distribution")


# ================================================================
# 7. DATASET CLASS
# ================================================================

class MRIImageDataset(Dataset):

    def __init__(self, image_paths, labels, transform):

        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):

        return len(self.image_paths)

    def __getitem__(self, index):

        image_path = self.image_paths[index]

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, self.labels[index], image_path


# ================================================================
# 8. LOAD PRETRAINED RESNET-50
# ================================================================

print("\nLoading ResNet-50...")

weights = models.ResNet50_Weights.DEFAULT

resnet = models.resnet50(weights=weights)

# Remove final classification layer
feature_extractor = nn.Sequential(
    *list(resnet.children())[:-1]
)

feature_extractor = feature_extractor.to(DEVICE)

feature_extractor.eval()

print("ResNet-50 feature extractor loaded.")


# ================================================================
# 9. EXTRACT DEEP FEATURES
# ================================================================

def extract_features(image_paths, labels):

    dataset = MRIImageDataset(
        image_paths,
        labels,
        transform
    )

    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0
    )

    all_features = []
    all_labels = []
    all_paths = []

    print("\nExtracting features...")

    with torch.no_grad():

        for batch_images, batch_labels, batch_paths in loader:

            batch_images = batch_images.to(DEVICE)

            features = feature_extractor(batch_images)

            features = features.view(
                features.size(0),
                -1
            )

            features = features.cpu().numpy()

            all_features.append(features)

            all_labels.extend(batch_labels)

            all_paths.extend(batch_paths)

    all_features = np.vstack(all_features)

    return all_features, all_labels, all_paths


# ================================================================
# 10. EXTRACT FEATURES FOR EACH DATASET
# ================================================================

train_features, train_labels, train_feature_paths = extract_features(
    train_paths,
    train_labels
)

val_features, val_labels, val_feature_paths = extract_features(
    val_paths,
    val_labels
)

test_features, test_labels, test_feature_paths = extract_features(
    test_paths,
    test_labels
)


print("\nFeature dimensions")

print("Train:", train_features.shape)
print("Validation:", val_features.shape)
print("Test:", test_features.shape)


# ================================================================
# 11. SAVE FEATURES
# ================================================================

np.save(
    os.path.join(OUTPUT_DIR, "train_features.npy"),
    train_features
)

np.save(
    os.path.join(OUTPUT_DIR, "validation_features.npy"),
    val_features
)

np.save(
    os.path.join(OUTPUT_DIR, "test_features.npy"),
    test_features
)


# ================================================================
# 12. NORMALIZE FEATURES
# ================================================================

train_features_norm = normalize(train_features)

val_features_norm = normalize(val_features)

test_features_norm = normalize(test_features)


# ================================================================
# 13. CREATE CLASS INDEX
# ================================================================

def create_class_index(labels):

    class_index = {}

    for index, label in enumerate(labels):

        if label not in class_index:
            class_index[label] = []

        class_index[label].append(index)

    return class_index


# ================================================================
# 14. GENERATE TRIPLETS
# ================================================================

def generate_triplets(
        features,
        labels,
        paths,
        number_of_triplets=15
):

    class_index = create_class_index(labels)

    available_classes = list(class_index.keys())

    valid_positive_classes = [
        c for c in available_classes
        if len(class_index[c]) >= 2
    ]

    if len(valid_positive_classes) < 2:

        raise ValueError(
            "At least two classes with two or more images "
            "are required for triplet generation."
        )

    triplets = []

    attempts = 0

    max_attempts = number_of_triplets * 100

    while (
        len(triplets) < number_of_triplets
        and attempts < max_attempts
    ):

        attempts += 1

        # Select anchor/positive class
        anchor_class = random.choice(
            valid_positive_classes
        )

        # Select negative class
        negative_classes = [
            c for c in available_classes
            if c != anchor_class
        ]

        if len(negative_classes) == 0:
            continue

        negative_class = random.choice(
            negative_classes
        )

        # Select anchor and positive
        anchor_idx, positive_idx = random.sample(
            class_index[anchor_class],
            2
        )

        # Select negative
        negative_idx = random.choice(
            class_index[negative_class]
        )

        triplets.append({

            "Anchor_Path":
                paths[anchor_idx],

            "Positive_Path":
                paths[positive_idx],

            "Negative_Path":
                paths[negative_idx],

            "Anchor_Label":
                labels[anchor_idx],

            "Positive_Label":
                labels[positive_idx],

            "Negative_Label":
                labels[negative_idx],

            "Anchor_Index":
                anchor_idx,

            "Positive_Index":
                positive_idx,

            "Negative_Index":
                negative_idx
        })

    return pd.DataFrame(triplets)


# ================================================================
# 15. CALCULATE TRIPLET DISTANCES
# ================================================================

def calculate_triplet_distances(
        triplet_df,
        features
):

    results = []

    for _, row in triplet_df.iterrows():

        anchor = features[
            int(row["Anchor_Index"])
        ]

        positive = features[
            int(row["Positive_Index"])
        ]

        negative = features[
            int(row["Negative_Index"])
        ]

        # ----------------------------------------------
        # Euclidean distances
        # ----------------------------------------------

        positive_distance = np.linalg.norm(
            anchor - positive
        )

        negative_distance = np.linalg.norm(
            anchor - negative
        )

        # ----------------------------------------------
        # Cosine similarity
        # ----------------------------------------------

        positive_similarity = cosine_similarity(
            anchor.reshape(1, -1),
            positive.reshape(1, -1)
        )[0][0]

        negative_similarity = cosine_similarity(
            anchor.reshape(1, -1),
            negative.reshape(1, -1)
        )[0][0]

        # ----------------------------------------------
        # Triplet margin
        # ----------------------------------------------

        distance_difference = (
            negative_distance -
            positive_distance
        )

        results.append({

            "Anchor_Path":
                row["Anchor_Path"],

            "Positive_Path":
                row["Positive_Path"],

            "Negative_Path":
                row["Negative_Path"],

            "Anchor_Label":
                row["Anchor_Label"],

            "Positive_Label":
                row["Positive_Label"],

            "Negative_Label":
                row["Negative_Label"],

            "Positive_Distance":
                positive_distance,

            "Negative_Distance":
                negative_distance,

            "Distance_Difference":
                distance_difference,

            "Positive_Cosine_Similarity":
                positive_similarity,

            "Negative_Cosine_Similarity":
                negative_similarity,

            "Triplet_Satisfied":
                negative_distance > positive_distance

        })

    return pd.DataFrame(results)


# ================================================================
# 16. FUNCTION FOR COMPLETE TRIPLET ANALYSIS
# ================================================================

def perform_triplet_analysis(
        features,
        labels,
        paths,
        dataset_name
):

    print("\n==============================================")
    print(dataset_name)
    print("==============================================")

    # Generate 15 triplets
    triplets_15 = generate_triplets(
        features,
        labels,
        paths,
        number_of_triplets=15
    )

    results_15 = calculate_triplet_distances(
        triplets_15,
        features
    )

    # Save all 15
    csv_file = os.path.join(
        OUTPUT_DIR,
        dataset_name + "_15_triplets.csv"
    )

    results_15.to_csv(
        csv_file,
        index=False
    )

    print("Saved:", csv_file)

    return results_15


# ================================================================
# 17. TRAIN TRIPLET ANALYSIS
# ================================================================

train_results = perform_triplet_analysis(
    train_features,
    train_labels,
    train_feature_paths,
    "Train"
)


# ================================================================
# 18. VALIDATION TRIPLET ANALYSIS
# ================================================================

if len(val_features) > 0:

    validation_results = perform_triplet_analysis(
        val_features,
        val_labels,
        val_feature_paths,
        "Validation"
    )


# ================================================================
# 19. TEST TRIPLET ANALYSIS
# ================================================================

if len(test_features) > 0:

    test_results = perform_triplet_analysis(
        test_features,
        test_labels,
        test_feature_paths,
        "Test"
    )


# ================================================================
# 20. GENERATE 3, 5 AND 15 TRIPLETS
# ================================================================

triplet_numbers = [3, 5, 15]

distance_results = {}


for n in triplet_numbers:

    print("\nGenerating", n, "triplets")

    triplets = generate_triplets(
        test_features,
        test_labels,
        test_feature_paths,
        number_of_triplets=n
    )

    results = calculate_triplet_distances(
        triplets,
        test_features
    )

    distance_results[n] = results

    output_csv = os.path.join(
        OUTPUT_DIR,
        f"Test_{n}_Triplets_Similarity.csv"
    )

    results.to_csv(
        output_csv,
        index=False
    )

    print(
        "Saved:",
        output_csv
    )


# ================================================================
# 21. TRIPLET DISTANCE MEASUREMENT
# ================================================================

plt.figure(figsize=(12, 7))

for n in triplet_numbers:

    results = distance_results[n]

    x = np.arange(
        1,
        len(results) + 1
    )

    plt.plot(
        x,
        results["Positive_Distance"],
        marker="o",
        linewidth=2,
        label=f"Anchor-Positive ({n} Triplets)"
    )

    plt.plot(
        x,
        results["Negative_Distance"],
        marker="s",
        linestyle="--",
        linewidth=2,
        label=f"Anchor-Negative ({n} Triplets)"
    )


plt.xlabel("Triplet Number")

plt.ylabel("Euclidean Distance")

plt.title(
    "Triplet Distance Measurement for ACL Knee MRI"
)

plt.legend()

plt.grid(True, alpha=0.3)

plt.tight_layout()

distance_plot = os.path.join(
    OUTPUT_DIR,
    "Triplet_Distance_Measurement_3_5_15.png"
)

plt.savefig(
    distance_plot,
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ================================================================
# 22. INDIVIDUAL PLOT FOR 3, 5 AND 15 TRIPLETS
# ================================================================

for n in triplet_numbers:

    results = distance_results[n]

    plt.figure(figsize=(10, 6))

    triplet_id = np.arange(
        1,
        len(results) + 1
    )

    plt.plot(
        triplet_id,
        results["Positive_Distance"],
        marker="o",
        linewidth=2,
        label="Anchor-Positive"
    )

    plt.plot(
        triplet_id,
        results["Negative_Distance"],
        marker="s",
        linewidth=2,
        linestyle="--",
        label="Anchor-Negative"
    )

    plt.xlabel("Triplet Number")

    plt.ylabel("Euclidean Distance")

    plt.title(
        f"Triplet Distance Measurement - {n} Triplets"
    )

    plt.legend()

    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    output_file = os.path.join(
        OUTPUT_DIR,
        f"Triplet_Distance_{n}_Triplets.png"
    )

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


# ================================================================
# 23. COSINE SIMILARITY PLOT
# ================================================================

plt.figure(figsize=(12, 7))

for n in triplet_numbers:

    results = distance_results[n]

    x = np.arange(
        1,
        len(results) + 1
    )

    plt.plot(
        x,
        results["Positive_Cosine_Similarity"],
        marker="o",
        linewidth=2,
        label=f"Anchor-Positive ({n})"
    )

    plt.plot(
        x,
        results["Negative_Cosine_Similarity"],
        marker="s",
        linestyle="--",
        linewidth=2,
        label=f"Anchor-Negative ({n})"
    )


plt.xlabel("Triplet Number")

plt.ylabel("Cosine Similarity")

plt.title(
    "Triplet Similarity Comparison for ACL Knee MRI"
)

plt.legend()

plt.grid(True, alpha=0.3)

plt.tight_layout()

similarity_plot = os.path.join(
    OUTPUT_DIR,
    "Triplet_Cosine_Similarity_3_5_15.png"
)

plt.savefig(
    similarity_plot,
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ================================================================
# 24. TRIPLET SATISFACTION
# ================================================================

summary = []

for n in triplet_numbers:

    results = distance_results[n]

    satisfied = results[
        "Triplet_Satisfied"
    ].sum()

    total = len(results)

    percentage = (
        satisfied / total
    ) * 100

    mean_positive_distance = (
        results["Positive_Distance"].mean()
    )

    mean_negative_distance = (
        results["Negative_Distance"].mean()
    )

    mean_positive_similarity = (
        results[
            "Positive_Cosine_Similarity"
        ].mean()
    )

    mean_negative_similarity = (
        results[
            "Negative_Cosine_Similarity"
        ].mean()
    )

    summary.append({

        "Number_of_Triplets":
            n,

        "Triplets_Satisfied":
            satisfied,

        "Total_Triplets":
            total,

        "Triplet_Satisfaction_Percentage":
            percentage,

        "Mean_Anchor_Positive_Distance":
            mean_positive_distance,

        "Mean_Anchor_Negative_Distance":
            mean_negative_distance,

        "Mean_Anchor_Positive_Similarity":
            mean_positive_similarity,

        "Mean_Anchor_Negative_Similarity":
            mean_negative_similarity
    })


summary_df = pd.DataFrame(summary)


# ================================================================
# 25. SAVE SUMMARY
# ================================================================

summary_file = os.path.join(
    OUTPUT_DIR,
    "Triplet_Distance_Similarity_Summary.csv"
)

summary_df.to_csv(
    summary_file,
    index=False
)


print("\n==============================================")
print("FINAL TRIPLET SUMMARY")
print("==============================================")

print(
    summary_df.to_string(index=False)
)

print("\nSummary saved to:")
print(summary_file)


# ================================================================
# 26. DISPLAY FIRST 15 TRIPLETS
# ================================================================

print("\n==============================================")
print("15 TRIPLET SIMILARITY RESULTS")
print("==============================================")

print(
    distance_results[15][[
        "Anchor_Label",
        "Positive_Label",
        "Negative_Label",
        "Positive_Distance",
        "Negative_Distance",
        "Positive_Cosine_Similarity",
        "Negative_Cosine_Similarity",
        "Triplet_Satisfied"
    ]].to_string(index=False)
)


print("\n==============================================")
print("PROCESS COMPLETED")
print("==============================================")

print("Results directory:")
print(OUTPUT_DIR)