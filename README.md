# Movie Recommender at Scale - MovieLens 27M (Spark ALS)

A production-style **collaborative filtering recommender** built with **Apache Spark MLlib (ALS)** on the **MovieLens 27M** dataset.  
Includes **end-to-end preprocessing**, **Spark-based EDA (Koalas)**, **ALS training**, and **both rating + Top-K ranking evaluation**. Bonus: plug in your **personal IMDb ratings** to generate truly personalized recommendations.

## What this does
- Loads MovieLens `ratings`, `movies`, and `links` into Spark
- Cleans, type-casts, and builds features required for ALS
- (Optional) merges **personal IMDb ratings** for a custom user profile
- Trains **ALS** for scalable recommendation generation
- Evaluates:
  - **Rating accuracy** (eg RMSE)
  - **Top-K quality** using a relevance threshold (**rating > 3**)

## Tech stack
- Python, **PySpark**, **Spark MLlib (ALS)**
- **Koalas** (Pandas-like EDA on Spark)
- Matplotlib

## Dataset
- **MovieLens 27M** (GroupLens): https://grouplens.org/datasets/movielens/  
  Expected files:
  - `movies.csv`
  - `ratings.csv`
  - `links.csv`
  - `Personal_IMBD_Ratings.csv` (optional)

## How to run

### Option A: Google Colab (recommended)
1. Open the notebook in Colab
2. Run setup cells to install:
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

## Pipeline (high level)
1. **Preprocessing**
   - Cast `userId`, `movieId` to int, `rating` to double
   - Join IMDb ids using `links.csv`
   - Merge optional personal ratings into the main ratings table
   - Create:
     - `ratingsScaled = rating - 2.5`
     - `ratingsBinary = 1 if ratingsScaled > 0 else 0`
2. **EDA**
   - Ratings-per-user distribution
   - Sparsity and cold-start insights
3. **Model training**
   - Spark MLlib **ALS**
   - Optional tuning via TrainValidationSplit / ParamGrid
4. **Evaluation**
   - Rating metrics (eg RMSE)
   - Top-K ranking metrics (precision/recall-style using relevance threshold)

## Outputs
- Trained ALS model (or best tuned model)
- Top movie recommendations per user
- Evaluation summary (rating + ranking metrics)
- EDA plots

## Notes / Tips
- Use `debug=True` to run fast on a smaller subset.
- Personal IMDb integration works best when IMDb ids match MovieLens `links.csv` format.
- Broadcast joins are disabled in the notebook for stability on large tables.

## License
For learning/academic use. MovieLens dataset is owned by GroupLens and follows their license terms.
