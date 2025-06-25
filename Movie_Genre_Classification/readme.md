## Introduction
The following program explores a multiclass classification problem in which predicting the movie genre based on external factors(title, description, content rating, rating, etc.) is the ultimate goal. SVM model is used to obtain both an efficent and accurate predictions.
<strong>The description is highly corelated with the final result and causes PERFECT results</strong>, which generally speaking is a red flag, but in this case, there is no data leakage between training and testing subsets. Vectorizer is used to form a BoW(bag of words)
approach aimed at converting strings into columns and further analysis of their appearances in descriptions/titles. 

## Case Study
<img src="plots/confussion.png" alt="confussion matrix" width="500">
<img src="plots/spread.png" alt="dist" width="500">
<img src="plots/average_rating.png" alt="average" width="500">
<img src="plots/ContentRating.png" alt="legend" width="500">
<img src="plots/g.png" alt="G" width="500">
<img src="plots/r.png" alt="R" width="500">
<img src="plots/pg13.png" alt="pg-13" width="500">
<img src="plots/nc17.png" alt="nc-17" width="500">
<img src="plots/pg.png" alt="pg" width="500">
