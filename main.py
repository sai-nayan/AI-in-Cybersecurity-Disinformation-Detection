import pandas as pd

# ==========================================
# STEP 1: LOAD THE DATASET
# ==========================================
# We load the dataset using pandas read_csv.
print("Loading the dataset...")
df = pd.read_csv("Features_For_Traditional_ML_Techniques.csv")

# Let's see the initial size (rows and columns)
print("Initial Dataset Shape (Rows, Columns):", df.shape)

# ==========================================
# STEP 2: REMOVE UNNECESSARY COLUMNS
# ==========================================
# 'Unnamed: 0' is just an old index column from CSV, we do not need it.
# 'embeddings' is a complicated text column, so as a beginner we can drop it.
columns_to_remove = ["Unnamed: 0", "embeddings"]
for col in columns_to_remove:
    if col in df.columns:
        df = df.drop(columns=[col])
        print(f"Removed column: {col}")

# ==========================================
# STEP 3: CLEAN COLUMN NAMES
# ==========================================
# Some columns have spaces or strange symbols like 'Word count' or 'TO’s'.
# We make them lowercase and replace spaces with underscores so they are easy to type.
df.columns = df.columns.str.strip().str.replace(" ", "_").str.replace("’", "").str.replace("'", "")
print("\nCleaned column names successfully!")

# ==========================================
# STEP 4: CHECK AND FIX MISSING VALUES
# ==========================================
# Check if there are any empty (null) cells in our data.
print("\nChecking for missing values:")
missing_values = df.isnull().sum()
total_missing = missing_values.sum()
print("Total missing values in dataset:", total_missing)

# If there were any missing values, we would fill them, but here there are 0!

# ==========================================
# STEP 5: CHECK AND REMOVE DUPLICATES
# ==========================================
# Check if any rows are repeated.
duplicates = df.duplicated().sum()
print("Number of duplicate rows found:", duplicates)
if duplicates > 0:
    df = df.drop_duplicates()
    print("Duplicates removed!")

# ==========================================
# STEP 6: BEGINNER-FRIENDLY ANALYSIS
# ==========================================
print("\n" + "=" * 50)
print("BEGINNER ANALYSIS (CYBERSECURITY & FAKE NEWS)")
print("=" * 50)

# 1. Target Distribution (1 = Real / Truth, 0 = Fake / Disinformation)
print("\n1. How many Real vs Fake posts are there?")
# 1 = Real news/tweets, 0 = Fake news/tweets
target_counts = df["BinaryNumTarget"].value_counts()
print(target_counts)
print("Percentage of Real (1):", round((target_counts[1.0] / len(df)) * 100, 2), "%")
print("Percentage of Fake (0):", round((target_counts[0.0] / len(df)) * 100, 2), "%")

# 2. Bot Accounts Analysis in Cybersecurity
print("\n2. How many accounts are bots vs human?")
# BotScoreBinary: 1 = Bot, 0 = Human
bot_counts = df["BotScoreBinary"].value_counts()
print(bot_counts)

# 3. Relationship: Are bots posting more Fake News?
print("\n3. Comparing Real vs Fake news across Bots and Humans:")
bot_vs_target = pd.crosstab(df["BotScoreBinary"], df["BinaryNumTarget"], normalize="index") * 100
print(bot_vs_target)
print("Explanation: Row 0 is Humans, Row 1 is Bots. Columns are 0 (Fake) and 1 (Real).")

# 4. Key Metrics comparison: Real vs Fake
# Compare average user credibility and follower count for Fake vs Real
print("\n4. Average Credibility and Followers for Fake (0) vs Real (1):")
comparison = df.groupby("BinaryNumTarget")[["cred", "followers_count", "retweets", "Word_count", "exclamation"]].mean()
print(comparison)

# ==========================================
# STEP 7: SAVE THE CLEANED DATASET
# ==========================================
# Save the cleaned data to a standard CSV file so we can use it for modeling
output_file = "cleaned_features_ml.csv"
print(f"\nSaving cleaned dataset to {output_file}...")
df.to_csv(output_file, index=False)
print("Done! Cleaned file is ready.")