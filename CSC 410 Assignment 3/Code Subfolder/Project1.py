#Logan Bowles python3
# -*- coding: utf-8 -*-

import pandas as pd
import cv2
import matplotlib.pyplot as plt   
import numpy as np                
import glob
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import Lasso
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Added from assignment 3
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
import seaborn as sns




############################################################################################################

# Step 4 function; simple code to find the original dimensions and resize it to meet assignment requirements
def image_resize(image_gray, height=256):
    
    # First ascertain the H x W of the image
    himg, wimg = image_gray.shape
    
    # Formula for aspect ratio and incorporating it to determine the width applicable
    aspect_ratio = wimg / himg
    new_width = int(height * aspect_ratio)
    
    # As per assignment, make sure that it is divisible by 8 (utlize remainder to do so)
    new_width = new_width - (new_width % 8)
    
    # Resize image and return it -> function complete
    resized_gray = cv2.resize(image_gray, (new_width, height), interpolation=cv2.INTER_AREA)
    
    return resized_gray



############################################################################################################

# Step 5 function, base of function directly from class notes, slightly augmented:

def myBlocks(img, label):
    
    # Ascertain the height and the width of the image (already resized)
    himg, wimg = img.shape
    
    # Count the number of 8 by 8 blocks that fit in the image exactly
    cellc = (himg // 8) * (wimg // 8)
    
    # Vector storage; intialize all as 0.  64 spots for data and 1 spot for result.
    flatc = np.zeros((cellc, 65), np.uint8)
    
    # Traverse through every block, left to right, then down, repeat until all traversed
    k = 0
    for i in range(0,himg,8):
        for j in range(0,wimg,8):
            
            # Extract an 8x8 block
            crop_tmp1 = img[i:i+8,j:j+8]
            
            # Vectorize / flatten the block
            flatc[k,0:64] = crop_tmp1.flatten()
            # Assign label to end
            flatc[k, 64] = label 
            
            # Move to next row
            k = k + 1
            
    # Return number of 8x8 blocks and flatc
    return cellc, flatc

############################################################################################################

# Step 6 function, similar to the one above obviously, just adjusted for sliding and overlap instead of moving by 8 perfectly 
 
def slideBlocks(img, label):
    
    block_size = 8
    slide = 4
    
    # Ascertain the height and the width of the image (already resized)
    himg, wimg = img.shape
    
    # Number of sliding windows that fit horizontally and vertically
    num_cells_vertical = (himg - block_size) // slide + 1
    num_cells_horizontal = (wimg - block_size) // slide + 1
    
    # Total number of sliding windows
    cellc = num_cells_vertical * num_cells_horizontal
    
    
    # Initialize array to store flattened feature vectors, add one for label
    flatc = np.zeros((cellc, block_size * block_size + 1), np.uint8)
    
    
    # Traverse through every block, left to right, then down, repeat until all traversed
    k = 0
    for i in range(0, himg - block_size + 1, slide):
        for j in range(0, wimg - block_size + 1, slide):
            
            # Extract an 8x8 block
            crop_tmp1 = img[i:i+block_size, j:j+block_size]
            
            # Vectorize / flatten the block
            flatc[k, 0:block_size * block_size] = crop_tmp1.flatten()
            # Assign label to end
            flatc[k, block_size * block_size] = label 
            
            # Move to next row
            k = k + 1
            
    # Return number of 8x8 blocks and flatc
    return cellc, flatc


############################################################################################################


# Step 7 code, just a really simply little function to print out information about csv files that are passed to it, in this instance for the feature vectors of pooling and sliding.

def csv_statistics(df, name):
    print(f"\n{name} Stats")
    print(f"Number of observations: {df.shape[0]}")
    print(f"Number of features: {df.shape[1]-1}")
    print("Rest of relevant information\n")
    print(df.describe())
    
    
############################################################################################################


# Step 7 code to print histograms as graph of choice.  Used inspiration from Chapter 3 in regards to visual and what it is showing.

def plot_histograms(df, name, feature_idx=32):

    plt.figure()
    plt.hist(df[f"feature_{feature_idx}"], bins=20, color='blue', edgecolor='black')
    plt.title(f"{name} - Feature 32 Distribution")
    plt.xlabel("Feature Value")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()




############################################################################################################


# To suffice for Task 2 and then Task 10, read all files from a folder and then print via a For loop 
folder_path = "/Users/loganbowles/Desktop/Task 2/Selected Fruits"

# Get all image files in the folder that are jpg / jpeg.  Utilizing glob: https://www.geeksforgeeks.org/how-to-use-glob-function-to-find-files-recursively-in-python/#google_vignette
image_files = glob.glob(folder_path + "/*.jpg")

# For counting iterations through the loop and making information easier to read.
image_counter = 1

# Initialize dataset storage for pooling effect
all_data = []

# Initialize dataset storage for convolution effect
all_data_sliding = []

# Loop through each image and display it via plts and cv2
for img_path in image_files:
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Display images unadulterated and then type shape to the console
    plt.imshow(img)
    plt.title("normal fruit")  
    plt.axis("off")
    plt.show()
    print(f"Image {image_counter}: \nShape: {img.shape}")
    
    image_counter += 1
    
    #Display different color channels (task 3) -> Code directly from Files in class; just adjusted for my names
    plt.imshow(img[:,:,0], cmap='Blues')
    plt.axis('off')
    plt.show()
    plt.imshow(img[:,:,1], cmap='Greens')
    plt.axis('off')
    plt.show()
    plt.imshow(img[:,:,2], cmap='Reds')
    plt.axis('off')
    plt.show()
    

    # Convert to grayscale. Also from class files
    image_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Call resize function "image_resize"; then print it and its dimensions
    resized_gray = image_resize(image_gray, height=256)
    plt.imshow(resized_gray, cmap='gray')
    plt.title("gray image")
    plt.axis("on")
    plt.show()
    print(f"New Shape: {resized_gray.shape}")
    
    
    # Finishing step 5
    # Generate feature vectors and add labels by calling myBlocks function.  Goal is to only extract the array so we ignore the cell count.  feature_vectors will be tbe list of all vectors
    _, feature_vectors = myBlocks(resized_gray, label=image_counter - 1)  
    
    # Convert to an array
    feature_vectors = np.array(feature_vectors)  
    
    # Using a dataframe to create a csv: https://www.geeksforgeeks.org/saving-a-pandas-dataframe-as-a-csv/
    df = pd.DataFrame(feature_vectors, columns=[f"feature_{i}" for i in range(64)] + ["label"])

    # Save to desired location
    csv_path = f"/Users/loganbowles/Desktop/Task 2/feature_vectors_{image_counter-2}.csv"
    df.to_csv(csv_path, index=False)
    
    # Printing the file location just so it is easy to validate
    print(f"Saved: {csv_path}\n")
    
    # Store this iteration to the end of the all_data array to merge after all images have been completed
    all_data.append(df)
    
    
    ###############################################################################################################
    
    # Finishing step 6, obviously the same logic, just save them with different file names.
    # Generate feature vectors and add labels again but this time by calling the slideBlocks function.  
    _, feature_vectors = slideBlocks(resized_gray, label=image_counter - 1)  
    
    # Convert to an array
    feature_vectors = np.array(feature_vectors)  
    
    # Using a dataframe to create a csv: https://www.geeksforgeeks.org/saving-a-pandas-dataframe-as-a-csv/
    df = pd.DataFrame(feature_vectors, columns=[f"feature_{i}" for i in range(64)] + ["label"])

    # Save to desired location
    csv_path = f"/Users/loganbowles/Desktop/Task 2/feature_vectors_slide_{image_counter-2}.csv"
    df.to_csv(csv_path, index=False)
    
    # Printing the file location just so it is easy to validate
    print(f"Saved: {csv_path}\n")
    
    # Store this iteration to the end of the all_data array to merge after all images have been completed
    all_data_sliding.append(df)
    
    
# Merge all of the dataframes that are currently in "all_data" into a single dataframe
merged_df = pd.concat(all_data, ignore_index=True)

# Shuffle the dataframe: https://www.geeksforgeeks.org/pandas-how-to-shuffle-a-dataframe-rows/
merged_df = merged_df.sample(frac=1)

# Save merged dataset
merged_csv_path = "/Users/loganbowles/Desktop/Task 2/merged_feature_vectors.csv"
merged_df.to_csv(merged_csv_path, index=False)

# Printing the file location just so it is easy to validate
print(f"Merged dataset saved: {merged_csv_path}")


# Merge and save sliding window feature vectors, same logic as above.
merged_sliding_df = pd.concat(all_data_sliding, ignore_index=True)
merged_sliding_df = merged_sliding_df.sample(frac=1)  
merged_sliding_csv = "/Users/loganbowles/Desktop/Task 2/merged_feature_vectors_sliding.csv"
merged_sliding_df.to_csv(merged_sliding_csv, index=False)
print(f"Merged sliding dataset saved: {merged_sliding_csv}")
############################################################################################################

# Step 7: Print statistical data ascertained so far.

# Read in the csv files to print information about pooling csvs
image0_df = pd.read_csv("/Users/loganbowles/Desktop/Task 2/feature_vectors_0.csv")
image1_df = pd.read_csv("/Users/loganbowles/Desktop/Task 2/feature_vectors_1.csv")
image2_df = pd.read_csv("/Users/loganbowles/Desktop/Task 2/feature_vectors_2.csv")

# Read in the csv file to print information about the sliding csvs
image0_slide_df = pd.read_csv("/Users/loganbowles/Desktop/Task 2/feature_vectors_slide_1.csv")
image1_slide_df = pd.read_csv("/Users/loganbowles/Desktop/Task 2/feature_vectors_slide_1.csv")
image2_slide_df = pd.read_csv("/Users/loganbowles/Desktop/Task 2/feature_vectors_slide_1.csv")

# Call function to print statiscal data for pooling
csv_statistics(image0_df, "image01_pooling")
csv_statistics(image1_df, "image02_pooling")
csv_statistics(image2_df, "image03_pooling")

# Call function to print statiscal data for sliding
csv_statistics(image0_slide_df, "image01_sliding")
csv_statistics(image1_slide_df, "image02_sliding")
csv_statistics(image2_slide_df, "image03_sliding")

print("\n")


# Plot histograms for a few selected feature columns
plot_histograms(image0_df, "Image01 Pooling")
plot_histograms(image1_df, "Image02 Pooling")
plot_histograms(image2_df, "Image03 Pooling")

plot_histograms(image0_slide_df, "Image01 Sliding")
plot_histograms(image1_slide_df, "Image02 Sliding")
plot_histograms(image2_slide_df, "Image03 Sliding")

############################################################################################################

# Step 8: Constructing a feature space


# Merge the csvs by stacking rows
image01_df = pd.concat([image0_df, image1_df], ignore_index=True)

# Shuffle the dataset (row order only)
image01_df = image01_df.sample(frac=1)  

# Save as image01.csv
image01_csv_path = "/Users/loganbowles/Desktop/Task 2/image01.csv"
image01_df.to_csv(image01_csv_path, index=False)

# Printing the file location just so it is easy to validate
print(f"Feature space for image01.csv saved: {image01_csv_path}")

# Move to next part of Step 8, attaining features from all three images.

# Merge image0, image1, and image2
image012_df = pd.concat([image0_df, image1_df, image2_df], ignore_index=True)

# Shuffle the dataset
image012_df = image012_df.sample(frac=1)

# Save as image012.csv
image012_csv_path = "/Users/loganbowles/Desktop/Task 2/image012.csv"
image012_df.to_csv(image012_csv_path, index=False)

print(f"Feature space for image012.csv saved at: {image012_csv_path} \n\n")


############################################################################################################


# Step 9: Generate subspaces and print ,ultidimensional scatterplots related 


# Load the dataset (Replace with your actual path)
df = pd.read_csv("/Users/loganbowles/Desktop/Task 2/image01.csv")

# Select two features for visualization
feature_x = "feature_0"
feature_y = "feature_1"

# Create scatter plot
plt.figure()
for label in df["label"].unique():
    subset = df[df["label"] == label]
    plt.scatter(subset[feature_x], subset[feature_y], label=f"Class {label}")

# Add labels and title
plt.xlabel(feature_x)
plt.ylabel(feature_y)
plt.title("2D Feature Space")
plt.legend()
plt.grid(True)
plt.show()

# Now create for three features.

# Select three features for 3D visualization
feature_x = "feature_0"
feature_y = "feature_1"
feature_z = "feature_2"

# Create 3D scatter plot
df = pd.read_csv("/Users/loganbowles/Desktop/Task 2/image012.csv")

fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(projection='3d')

for label in df["label"].unique():
    subset = df[df["label"] == label]
    ax.scatter(subset[feature_x], subset[feature_y], subset[feature_z], label=f"Class {label}", alpha=0.7)

# Add labels and title
ax.set_xlabel(feature_x)
ax.set_ylabel(feature_y)
ax.set_zlabel(feature_z)
ax.set_title("3D Feature Space")
ax.legend()
plt.show()


############################################################################################################

# Beginning Assignment 2 - Task 1, part 1 80/20 most of code from class-code-ridge-reg.txt
print("###############\nAssignment 2 - Task 1\n")

# Read a feature space
input_data = pd.read_csv("/Users/loganbowles/Desktop/Task 2/image01.csv",header=None)

# Last column is label
NN = 64

# Label/Response set
y = input_data[NN]

# Drop the labels and store the features
X = input_data.drop(NN, axis=1)

# Convert to NumPy arrays:
#Feature Matrix
X1 = X.to_numpy() 
#Label Vector
Y1 = y.to_numpy()

# Split the data into 80:20
row, col = X1.shape

# Number of training samples
TR = round(row*0.8)

# Training with 80%
X1_train = X1[:TR,:]
Y1_train = Y1[:TR]

# Testing with 20%
X1_test = X1[TR:row,:]
Y1_test = Y1[TR:row]

# Save datasets to CSV files
train_features_path = "/Users/loganbowles/Desktop/Task 2/X_train.csv"
train_labels_path = "/Users/loganbowles/Desktop/Task 2/Y_train.csv"
test_features_path = "/Users/loganbowles/Desktop/Task 2/X_test.csv"
test_labels_path = "/Users/loganbowles/Desktop/Task 2/Y_test.csv"

pd.DataFrame(X1_train).to_csv(train_features_path, index = False, header = False)
pd.DataFrame(Y1_train).to_csv(train_labels_path, index = False, header = False)
pd.DataFrame(X1_test).to_csv(test_features_path, index = False, header = False)
pd.DataFrame(Y1_test).to_csv(test_labels_path, index = False, header = False)

print(f"Training features saved to {train_features_path}")
print(f"Training labels saved to {train_labels_path}")
print(f"Test features saved to {test_features_path}")
print(f"Test labels saved to {test_labels_path}")


# Task 1, Part 2 

# Load the training and testing sets
X_train = pd.read_csv("/Users/loganbowles/Desktop/Task 2/X_train.csv", header = None)
X_test = pd.read_csv("/Users/loganbowles/Desktop/Task 2/X_test.csv", header = None)

# Selecting two features
feature_1 = 0
feature_2 = 1  

#print(X_train.dtypes); Continued getting error: 'Could not convert string '{x}' to numeric'.  It was showing all cells as objects rather than numbers.
#https://pandas.pydata.org/docs/reference/api/pandas.to_numeric.html to resolve this without it filling the outputs to console.

X_train = X_train.apply(pd.to_numeric, errors='coerce')
X_test = X_test.apply(pd.to_numeric, errors='coerce')

# Compute mean and variance
train_mean_1, train_var_1 = X_train[feature_1].mean(), X_train[feature_1].var()
test_mean_1, test_var_1 = X_test[feature_1].mean(), X_test[feature_1].var()

train_mean_2, train_var_2 = X_train[feature_2].mean(), X_train[feature_2].var()
test_mean_2, test_var_2 = X_test[feature_2].mean(), X_test[feature_2].var()

# Get shape of the data
train_shape_1 = X_train[feature_1].shape
test_shape_1 = X_test[feature_1].shape

train_shape_2 = X_train[feature_2].shape
test_shape_2 = X_test[feature_2].shape

# Print statistics including shape
print(f"\nFeature {feature_1}:")
print(f"Train Mean={train_mean_1:.2f}, Train Var={train_var_1:.2f}, Train Shape={train_shape_1}")
print(f"Test Mean={test_mean_1:.2f}, Test Var={test_var_1:.2f}, Test Shape={test_shape_1}\n")

print(f"Feature {feature_2}:")
print(f"Train Mean={train_mean_2:.2f}, Train Var={train_var_2:.2f}, Train Shape={train_shape_2}")
print(f"Test Mean={test_mean_2:.2f}, Test Var={test_var_2:.2f}, Test Shape={test_shape_2}\n")

# Plot histogram in the same figure.  Utilizing 'density for the first time'. https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.hist.html was really helpful for more advanced plotting.
plt.figure(figsize=(10, 5))

# Histogram for Feature 1.  30 bars and so it is actually readable, I set opacity to half, left histogram
plt.subplot(1, 2, 1)
plt.hist(X_train[feature_1], bins=30, alpha=0.5, label="Train", density=True)
plt.hist(X_test[feature_1], bins=30, alpha=0.5, label="Test", density=True)
plt.xlabel(f"Feature {feature_1}")
plt.ylabel("Density")
plt.title(f"Histogram of Feature {feature_1}")
plt.legend()

# Histogram for Feature 2
plt.subplot(1, 2, 2)
plt.hist(X_train[feature_2], bins=30, alpha=0.5, label="Train", density=True)
plt.hist(X_test[feature_2], bins=30, alpha=0.5, label="Test", density=True)
plt.xlabel(f"Feature {feature_2}")
plt.ylabel("Density")
plt.title(f"Histogram of Feature {feature_2}")
plt.legend()
plt.show()

# Task 1, Part 3 

# Load training labels
y_train = pd.read_csv("/Users/loganbowles/Desktop/Task 2/Y_train.csv", header=None) 
y_test = pd.read_csv("/Users/loganbowles/Desktop/Task 2/Y_test.csv", header=None) 

plt.figure(figsize=(10, 5))

# Scatter plot for Training Set
plt.subplot(1, 2, 1)
for label in y_train[0].unique():
    subset = X_train[y_train[0] == label]
    plt.scatter(subset[feature_1], subset[feature_2], label=f"Class {label}", alpha=0.6)

plt.xlabel(f"Feature {feature_1}")
plt.ylabel(f"Feature {feature_2}")
plt.title("Scatter Plot of Training Data")
plt.legend()
plt.grid(True)

# Scatter plot for Testing Set
plt.subplot(1, 2, 2)
for label in y_test[0].unique():
    subset = X_test[y_test[0] == label]
    plt.scatter(subset[feature_1], subset[feature_2], label=f"Class {label}", alpha=0.6)

plt.xlabel(f"Feature {feature_1}")
plt.ylabel(f"Feature {feature_2}")
plt.title("Scatter Plot of Testing Data")
plt.legend()
plt.grid(True)

plt.show()


############################################################################################################

# Task 2 - Most of this is directly from the class notes
print("###############\nAssignment 2 - Task 2\n")

# Read a feature space
input_data = pd.read_csv("/Users/loganbowles/Desktop/Task 2/image01.csv")

# Label/Response set
y = input_data['label']

# Drop the labels and store the features
X = input_data.drop('label', axis=1)

# Convert to NumPy arrays:
#Feature Matrix
X1 = X.to_numpy() 
#Label Vector
Y1 = y.to_numpy()

# Split the data into 80:20
row, col = X1.shape

# Number of training samples
TR = round(row*0.8)

# Training with 80%
X1_train = X1[:TR,:]
Y1_train = Y1[:TR]

# Testing with 20%
X1_test = X1[TR:row,:]
Y1_test = Y1[TR:row]

# Convert to NumPy to avoid 'X does not have feature names but Lasso was fitted with Feature Names'
X1_train = np.array(X1_train)  
X1_test = np.array(X1_test)

# Recieved direct error message telling me to up the iterations due to convergence not being achieved.
reg = Lasso(alpha=0.08, max_iter=10000)
model = reg.fit(X1_train, Y1_train)

yhat_test = model.predict(X1_test)
yhat_test = yhat_test.round()

# Confusion matrix analytics
CC_test = confusion_matrix(y_test, yhat_test, labels=[0,1])

TN = CC_test[1,1]
FP = CC_test[1,0]
FN = CC_test[0,1]
TP = CC_test[0,0]

FPFN = FP+FN
TPTN = TP+TN

print("\nLasso: \n")


# Changed math, due to inaccurate answers and added if/else statements for divide by 0
Accuracy = 1/(1+(FPFN/TPTN)) if TPTN != 0 else 0
print("Our_Accuracy_Score:",Accuracy)

Precision = 1/(1+(FP/TP)) if TP != 0 else 0
print("Our_Precision_Score:",Precision)

Sensitivity = 1/(1+(FN/TP)) if TP != 0 else 0
print("Our_Sensitivity_Score:",Sensitivity)

Specificity = 1/(1+(FP/TN)) if TN != 0 else 0
print("Our_Specificity_Score:",Specificity, "\n")

# Literally nothing changed besides formatting 
print("Fixed Accuracy Score:", Accuracy)
print("Fixed Precision Score:", Precision)
print("Fixed Sensitivity Score:", Sensitivity)
print("Fixed Specificity Score:", Specificity, "\n")

# Built-in accuracy measure from sklearn import metrics
print("BuiltIn_Accuracy:",metrics.accuracy_score(y_test, yhat_test))
print("BuiltIn_Precision:",metrics.precision_score(y_test, yhat_test))
print("BuiltIn_Sensitivity (recall):",metrics.recall_score(y_test, yhat_test), "\n")


# Task 2 part 2

# Model to make predictions
yhat_test = model.predict(X1_test)

# Round to 0 or 1 for classification
yhat_test = yhat_test.round()  

# Got error ' Data must be 1-dimensional, got ndarray of shape (512, 1) instead' from pd.dataframe, therefore have to flatten so I can do steps in pt 3
y_test = np.array(y_test).flatten()

# Combine actual and predicted labels into a DataFrame
results_df = pd.DataFrame({'Actual': y_test, 'Predicted': yhat_test})

# View the results
print(results_df.head(), "\n")

test_results_path = "/Users/loganbowles/Desktop/Task 2/test_results.csv"

results_df.to_csv(test_results_path, index=False)

print(f"Test Reults for test_results.csv saved at: {test_results_path}\n")


# Task 2 part 3

# confusion matrix construction
conf_matrix = confusion_matrix(results_df["Actual"], results_df["Predicted"])

# Convert to a DataFrame
conf_matrix_df = pd.DataFrame(conf_matrix, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"])

# Display the confusion matrix
print("Confusion Matrix:\n", conf_matrix_df, "\n")

# Save csv
conf_matrix_path = "/Users/loganbowles/Desktop/Task 2/confusion_matrix.csv"
conf_matrix_df.to_csv(conf_matrix_path)

print(f"Confusion matrix saved at: {conf_matrix_path} \n\n")

    
############################################################################################################

print("###############\nAssignment 2 - Task 3\n")


# 2-classes
input_data = pd.read_csv("/Users/loganbowles/Desktop/Task 2/image01.csv")

# Ascertain X, Y https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.iloc.html.  Much easier than what I was using before.
y = input_data.iloc[:, -1]  
X = input_data.iloc[:, :-1]  

# Very useful!: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html

# Train/Test Split (80/20) -> test_size = 0.2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Random Forest implementation (100 trees)
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)

# Predictability, actually test
rf_preds = rf.predict(X_test)

# Save results
results_rf = pd.DataFrame({'Actual': y_test, 'Predicted': rf_preds})
results_rf.to_csv("/Users/loganbowles/Desktop/Task 2/rf_twoclass_results.csv", index=False)

# Save confusion matrix (task 3 - Step 3) - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
conf_rf = confusion_matrix(y_test, rf_preds)
pd.DataFrame(conf_rf).to_csv("/Users/loganbowles/Desktop/Task 2/rf_twoclass_confusion_matrix.csv", index=False)
print("Random Forest (2-class) \n", classification_report(y_test, rf_preds))


# Literally the same thing but with three classes (different csv file) and no comments 

# 3-classes
input_data = pd.read_csv("/Users/loganbowles/Desktop/Task 2/image012.csv")

y = input_data.iloc[:, -1]  
X = input_data.iloc[:, :-1]  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)

results_rf = pd.DataFrame({'Actual': y_test, 'Predicted': rf_preds})
results_rf.to_csv("/Users/loganbowles/Desktop/Task 2/rf_multiclass_results.csv", index=False)

conf_rf = confusion_matrix(y_test, rf_preds)
pd.DataFrame(conf_rf).to_csv("/Users/loganbowles/Desktop/Task 2/rf_multiclass_confusion_matrix.csv", index=False)
print("Random Forest (3-class) \n", classification_report(y_test, rf_preds))

##################################################################################################

print("###############\nAssignment 3\n")


# Most very similar to Assignment 2, so I will not overinundate with new comments purveying the same thing
# Other important stuff is from the related class file to PCA 

# Load the 2-class dataset
input_data = pd.read_csv("/Users/loganbowles/Desktop/Task 2/image01.csv")
y = input_data.iloc[:, -1]
X = input_data.iloc[:, :-1]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Standardize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply PCA
cmp = 32
pca = PCA(n_components=cmp)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Train Random Forest with OOB
rf = RandomForestClassifier(random_state=0, n_estimators=500, oob_score=True, n_jobs=-1)
rf.fit(X_train_pca, y_train)

# Evaluate OOB error
oob_error = 1 - rf.oob_score_
print("oob error", "\n", {oob_error}, "\n\n")

# Predict
y_preds = rf.predict(X_test_pca)

# confusion matrix construction
conf_matrix = confusion_matrix(results_df["Actual"], results_df["Predicted"])

# Convert to a DataFrame
conf_matrix_df = pd.DataFrame(conf_matrix, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"])

# Display the confusion matrix
print("Confusion Matrix:\n", conf_matrix_df, "\n")

# Save csv
conf_matrix_path = "/Users/loganbowles/Desktop/Task 2/confusion_matrix.csv"
conf_matrix_df.to_csv(conf_matrix_path)

print(f"Confusion matrix saved at: {conf_matrix_path} \n\n")

# Accuracy Score
accuracy = accuracy_score(y_test, y_preds)
print(f"Accuracy: {accuracy:.4f}\n")



