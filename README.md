# Spotify Songs Analysis

### Problem Definition

The global music industry was worth over $19 billion in 2018, and there has been recent interest in predicting the popularity of a song based solely on its musical features. Through our research, we found that there have been many similar studies conducted on music and machine learning in recent years. We found studies that trained CRNNs on a combination of the time-series waveform data and musical features data, as well as studies that trained SVM models on features extracted from the songs [2, 3, 5]. In this project, however, we plan to test if machine learning models can make accurate predictions using the musical features supplied by the Spotify Dataset. We believe that this approach has potential since there has not been an experiement like this before: past projects based on musical features only used simple classifiers like SVM and past projects using neural networks focused more on the waveform data. We believe that by applying more complex ML models to the musical feature data, we may be able to extract useful patterns and insights.


### Data Exploration

We got our dataset from the Spotify API, which provides its own generated musical features for each track. With each API call, we can collect the musical features for up to 100 songs, however since this would take some time to collect a large enough dataset, we used a preassembled dataset created using the same process. The dataset we used to train our models contains 228,159 tracks, and provides numerical data on the musical features of each song. Each track is identified by a tuple of (Artist Name, Track Name, Genre, Track ID) and has 10 musical features associated with it. After removing all identifying features, the dataset looks as follows:


![Data Preview](data_head.png)


We have also provided a brief explanation for each feature:


| Feature          | Explanation                                                                                                                                             |
|------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------|
| Popularity       | Based, in the most part, on the total number of plays the track has had and how recent those plays are.                                                 |
| Acousticness     | A confidence measure from 0.0 to 1.0 of whether the track is acoustic.                                                                                  |
| Danceability     | How suitable a track is for dancing based on a combination of musical elements including tempo, rhythm stability, beat strength, and overall regularity |
| Duration         | Duration in milliseconds.                                                                                                                               |
| Energy           | Energy is a measure from 0.0 to 1.0 and represents a perceptual measure of intensity and activity.                                                      |
| Instrumentalness | Predicts whether a track contains no vocals.                                                                                                            |
| Liveness         | Detects the presence of an audience in the recording.                                                                                                   |
| Loudness         | The overall loudness of a track in decibels (dB).                                                                                                       |
| Speechiness      | Detects the presence of spoken words in a track.                                                                                                        |
| Tempo            | The overall estimated tempo of a track in beats per minute (BPM).                                                                                       |
| Valence          | Describes the musical positiveness conveyed by a track.   


After standardizing the musical feature data subset, we performed PCA to better understand the underlying data's core composition. It turned out that the explained variance was reasonably well distributed amongst the various musical features with 99% variance explanation requiring 8 of 9 principal components and only ~40% explained by the first component. We capped the number of components at 9 since there were only 9 musical features under consideration. 

<p align="center"> 
<img src="dbscan_images/musical_feature_pca_explained_variance.png">
</p>

Interestingly, the primary component was best described by loudness and energy, which a random sampling of adults above the age of 45 has subsequently confirmed via their ground truth labels.

<p align="center"> 
<img src="dbscan_images/pca_feature_weights.png">
</p>

In order to get another view at what features may end up being the best predictors of popularity, we plotted each feature against popularity and looked for a positive correlation trend in the resulting plots. The features that ended up having the most highly correlated shape were Loudness and Danceability.

![Correlation Plots](corr_plots.png)


### Unsupervised

The goal of the DBSCAN unsupervised learning portion was to generate playlists comprised of tracks with similar musical features. To this end, we separated the dataset into the core numerical musical features taking care to filter out identifying labels and classifications such as the track name, artist, and genre. We also removed effectively duplicate tracks based on artist and track name, which reduced the dataset size from ~280,000 to ~150,000 tracks. The resultant list of musical features consisted of acousticness, danceability, energy, instrumentalness, liveness, loudness, speechiness, tempo, and valence.

In order to perform the DBSCAN clustering, we needed to determine relevant values for the  𝑚𝑖𝑛𝑝𝑡𝑠  and  𝜖  variables. We used the  𝑚𝑖𝑛𝑝𝑡𝑠≤𝐷+1  rule of thumb to set  𝑚𝑖𝑛𝑝𝑡𝑠  equal to 20 given that our cleaned dataset consisted of 9 features and we wanted to generate playlists with a sufficient number of tracks. We then leveraging the tried and true "elbow test" by plotting the sorted 10th nearest neighbor distances. Based on the elbow test, we elected to use an  𝜖  value of 0.65 to ensure both a sufficient number of clusters as well as a relatively evenly disbursed track count per cluster.

<p align="center"> 
<img src="dbscan_images/knn_distance.png">
</p>

The resultant clustering consisted of 14 playlists, excluding tracks labeled as noise, comprised of ~74% of the total dataset. The average cluster size consisted of ~8,735 tracks, but this was heavily dominated by a few outsized playlists. We further whittled down the resultant clustering to only those consisting of between 10 and 100 tracks, which generated our final 10 playlists.

Although our final playlists represented a fraction of the original dataset, the quality was relatively robust as evidenced by the proportion of the most prevalent genre in each playlist.

<p align="center"> 
<img src="dbscan_images/genre_distribution.png">
</p>

Without further adieu, below please find a sample of tracks from some of our DBSCAN generated playlists. Interestingly, despite leaving out the genre and artist name from the cleaned dataset, the DBSCAN clustering grouped together tracks from similar genres and artists based solely on their musical features. With additional musical features and/or the addition of millions of user "likes", we undoubtedly would have a larger, deeper set of available playlists.

<p align="center"> 
<img src="dbscan_images/playlists.PNG">
</p>

### Supervised

The first step was to take a look at all the features and construct a correlation matrix
to determine if any of the features represented the same meaning as another feature. This matrix is shown below. Energy and loudness are highly correlated as one might expect. All of the features are distinct from each other and will all be useful in the final regression models to predict popularity.

<p align="center"> 
<img src="images/correlation_matrix.PNG">
</p>

Four different regression models were constructed using scikit-learn. These models were 
standard linear regression, k-nearest neighbors, random forest and decision trees. All models were trained and scored using 10-fold cross validation. Two different scoring metrics were looked at, R squared value and root mean square error. These scores for each model are shown below. Hyperparameter tuning was performed on all of the models to get the best possible scores for each. Random forest ended up performing significantly better than all of the other models. 

<p align="center"> 
<img src="images/r2_bar_plot.png">
</p>

<p align="center"> 
<img src="images/rmse_bar_plot.png">
</p>

The next step was to look at all of the features that went into the models are see which feature ended up being the most useful for the model. This was done by looking at scikit-learn's feature importance variable and it ended up being that acousticness was by far the most useful to the model. Many of the other features were relatively similar in importance to each other and still played a significant role in producing the model. The graph below shows the top 10 features. The two features that played almost no role in the model were key and time_signature. 

<p align="center"> 
<img src="images/feature_importances_random_forest.png">
</p>

### Future Work

The next steps for this project will be to get a larger dataset, and find new song features that would help to improve the predictive performance of our models. The biggest source of new information we foresee is using song audio data and analyzing that with a deep neural network. Similar work has been done using audio data for song genre classification [6]. We can apply the same network and use the features it extracts from the audio data for popularity prediction rather than genre classification. 


### References

[1] Bertin-Mahieux, T., Ellis, D. P., Whitman, B., & Lamere, P. (2011). The million song dataset.

[2] Choi, K., Fazekas, G., Sandler, M., & Cho, K. (2017). Convolutional recurrent neural
networks for music classification. In IEEE International Conference on Acoustics,
Speech and Signal Processing (ICASSP) (pp. 2392-2396).

[3] J. Lee and J. Lee, "Music Popularity: Metrics, Characteristics, and Audio-Based Prediction," in IEEE Transactions on Multimedia, vol. 20, no. 11, pp. 3173-3182, Nov. 2018.

[4] McFee, B., Bertin-Mahieux, T., Ellis, D. P., & Lanckriet, G. R. (2012). The million song dataset challenge. In Proceedings of the 21st International Conference on World Wide 
Web (pp. 909-916). ACM.

[5] Ni, Y., Santos-Rodriguez, R., Mcvicar, M., & De Bie, T. (2011). Hit song science 
once again a science. In 4th International Workshop on Machine Learning and Music.

[6] Yu, Yang et al. Deep attention based music genre classification. Neurocomputing, vol 372, pp. 84-91.  Jan. 2020.


Current URL: https://deom119.github.io/deom119.github.io-cs7641/
