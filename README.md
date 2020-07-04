# Predicting Pulsars using the HTRU 2 Data set

To learn about pulsars, watch these two videos-

- https://www.youtube.com/watch?v=gjLk_72V9Bw -This is by NASA's Goddard Space Center
- (https://www.youtube.com/watch?v=bKkh7viXjqs) - By Astronomate.  

I would **highly** recommend that you read this post:
- [https://as595.github.io/classification/](https://as595.github.io/classification/)
## Sample(of size 4) of the data used

| #   | Mean of the integrated profile | Standard deviation of the integrated profile | Excess kurtosis of the integrated profile | Skewness of the integrated profile | Mean of the DM-SNR curve | Standard deviation of the DM-SNR curve | Excess kurtosis of the DM-SNR curve | Standard deviation of the DM-SNR curve | Skewness of the DM-SNR curve | target_class_ |
| --- | ------------------------------ | -------------------------------------------- | ----------------------------------------- | ---------------------------------- | ------------------------ | -------------------------------------- | ----------------------------------- | -------------------------------------- | ---------------------------- | ------------- |
| 0   | 140.562500                     | 55.683782                                    | -0.234571                                 | -0.699648                          | 3.199833                 | 19.110426                              | 7.975532                            | 74.242225                              | 74.242225                    | 0             |
| 1   | 102.507812                     | 58.882430                                    | 0.465318                                  | -0.515088                          | 1.677258                 | 14.860146                              | 10.576487                           | 127.393580                             | 127.393580                   | 0             |
| 2   | 103.015625                     | 39.341649                                    | 0.323328                                  | 1.051164                           | 3.121237                 | 21.744669                              | 7.735822                            | 63.171909                              | 63.171909                    | 0             |
| 3   | 136.750000                     | 57.178449                                    | -0.068415                                 | -0.636238                          | 3.642977                 | 20.959280                              | 6.896499                            | 53.593661                              | 53.593661                    | 0             |
| 4   | 88.726562                      | 40.672225                                    | 0.600866                                  | 1.123492                           | 1.178930                 | 11.468720                              | 14.269573                           | 252.567306                             | 252.567306                   | 0             |

## Designing the model
In this project, I decided to create a very simple Logistic Regression model (no ResNets be seen) that classifies pulsars.  
### Ingest Data
![Gif on Giphy](https://media.giphy.com/media/l2QDSUB9ToTh8mucM/giphy-downsized.gif)  

The **easiest** way to do this would be to add [this dataset](https://www.kaggle.com/pavanraj159/predicting-a-pulsar-star) on Kaggle

<iframe  title="YouTube video player" width="480" height="390" src="http://www.youtube.com/watch?v=clWICXLpnMo?autoplay=1" frameborder="0" allowfullscreen></iframe>

# Credits

R. J. Lyon, B. W. Stappers, S. Cooper, J. M. Brooke, J. D. Knowles,
Fifty Years of Pulsar Candidate Selection: From simple filters to a new
principled real-time classification approach, Monthly Notices of the
Royal Astronomical Society 459 (1), 1104-1123, DOI: 10.1093/mnras/stw656
## Links
- You can use an "image" data set called [HTRU1](https://github.com/as595/HTRU1) to achieve the same results(It also has a greater number of entry points at 60,000).
- An easier way to implement this would be to use SKLearn. This would also allow you to use KNN and SVN(and other classifier models) really easily. [Check out this awesome data set on Kaggle](https://www.kaggle.com/ytaskiran/predicting-class-of-pulsars-with-ml-algorithms)
