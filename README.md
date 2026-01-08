# Collaborative Filtering Recommender System (MovieLens 27M) - Spark ALS

A scalable movie recommender built on **Apache Spark MLlib** using **collaborative filtering (ALS)**.  
This project covers **data preprocessing**, **EDA (Koalas / Pandas-like on Spark)**, **model training**, and **recommendation quality evaluation** on the **MovieLens 27M** ratings dataset.

## What this does
- Loads MovieLens ratings, movies, and links tables into Spark
- Cleans and type-casts columns needed for ALS
- (Optional) Merges **personal IMDb ratings** into the MovieLens ratings table for a custom user profile
- Trains an **ALS** model for implicit-style ranking + rating prediction
- Evaluates both **rating accuracy** and **top-K ranking quality**
  - A movie is treated as *relevant* if **rating > 3** (binarized target)

## Tech stack
- Python, PySpark, Spark MLlib (ALS)
- Koalas (Pandas-like EDA on Spark)
- Matplotlib (charts)

## Dataset
- **MovieLens 27M** ratings dataset  
  Expected files:
  - `movies.csv`
  - `ratings.csv`
  - `links.csv`
  - `Personal_IMBD_Ratings.csv` (optional)

## How to run

### Option A: Google Colab (recommended)
1. Open the notebook in Colab.
2. Run the setup cells to install:
   - Java 8
   - Spark (2.4.x)
   - findspark
   - koalas
3. Mount Google Drive and set:
   - `DATA_PATH` -> folder containing dataset CSVs  
   - `RESULTS_PATH` -> folder to write outputs

### Option B: Local Spark
Requirements:
- Java 8+ (Spark-compatible)
- Apache Spark installed and `SPARK_HOME` set
- Python packages: `pyspark`, `findspark`, `koalas` (optional)

Run the notebook normally after updating paths.

## Key steps (pipeline)
1. **Preprocessing**
   - Cast `userId`, `movieId` to int, `rating` to double
   - Join IMDb ids using `links.csv`
   - Merge optional personal ratings into the main rating table
   - Create:
     - `ratingsScaled = rating - 2.5`
     - `ratingsBinary = 1 if ratingsScaled > 0 else 0`
2. **EDA**
   - Distribution of number of ratings per user
   - Basic stats useful for sparsity and cold-start intuition
3. **Model training**
   - ALS from Spark MLlib
   - Tuned via TrainValidationSplit / ParamGrid (if enabled)
4. **Evaluation**
   - Rating prediction metrics (ex: RMSE)
   - Ranking metrics on Top-K (ex: precision/recall-style metrics using relevant threshold)

## Outputs
- Trained ALS model (and/or best model from tuning)
- Recommendation samples (top movies per user)
- Evaluation metrics summary
- EDA figures (rating distribution plots)

## Notes / Tips
- For quick testing, enable the notebook `debug=True` mode to limit rows.
- If you use personal IMDb ratings, ensure the IMDb ids match MovieLens link format.
- Spark settings in the notebook disable auto broadcast joins for stability on large tables.

## License
For academic/learning use. Dataset belongs to the MovieLens authors and follows their license terms.
