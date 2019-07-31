## Exercise derived from https://www.dataquest.io/blog/data-science-project-fandango/ with modifications
# I expanded the analysis by including joining data frames, running a permutation test and visualizing the data.


import pandas as pd
fandango_score = pd.read_csv('fandango_score_comparison.csv')
movie_ratings = pd.read_csv('movie_ratings_16_17.csv')
print(fandango_score.head(5))
print(movie_ratings.head(5))
# use .shape to learn the shape of data
fandango_score.shape
new_columns = []
# for loop to capture all the columns with fandango in the name
for subject in fandango_score.columns:
	if 'Fandango' in subject:
		new_columns.append(subject)
# add FILM to the first column
new_columns.insert(0,'FILM')
fandango_previous = fandango_score[new_columns].copy()

fandango_after = movie_ratings[['movie','year','fandango']].copy()

import matplotlib.pyplot as plt
fan_reviews = plt.hist(fandango_previous['Fandango_votes'], range=[0,100], bins = 20)
# set the x range between 0 and 100 only since we only care if there're reviews < 30
plt.axvline(30, color = 'r', linestyle = 'dashed')
# we also want to see how the data distributed around our cutoff line x = 30
plt.show()
# confirm that all movies here have votes 30 or more votes
under_30 = sum(fandango_previous['Fandango_votes'] < 30)
# add the column year by extracting year info from FILM
fandango_previous['year'] = fandango_previous['FILM'].str[-5:-1]
# make a new table with only movies from 2015 and 2016. Note 2016 in fandango_after is an integer. 
fandango_2015 = fandango_previous[fandango_previous['year']=='2015'].copy()
fandango_2016 = fandango_after[fandango_after['year'] == 2016].copy()
fandango_2015.shape
fandango_2016.shape
# 129 movies in 2015 and 191 movies in 2016
# changing the year data in 2015 table from str to integer so I can join the 2015 and 2016 data frames
import numpy as np
fandango_2015.year = pd.to_numeric(fandango_2015.year, errors = 'coerce').astype(np.int64)

fandango_2016.columns = ['FILM','year','Fandango_Stars']
# use concat function to merge 2 dataframes
fandango_15_and_16 = pd.concat([fandango_2015,fandango_2016], join = 'outer',sort=True)

#actual average ratings
total_average = fandango_15_and_16.Fandango_Stars.mean()
## Conduct a permutation test to determine if there is a difference in the average ratings from 2015 and 2016
#randomly select 129 samples and assign them as ratings from 2015
fifteen_avg_list = []
sixteen_avg_list = []
for i in range(10000):
	random15 = fandango_15_and_16.sample(129)
	fifteen_avg_list.append(random15.Fandango_Stars.mean())
	sixteen_avg_list.append((fandango_15_and_16.Fandango_Stars.sum()-random15.Fandango_Stars.sum())/191)

# calculate all the mean differences from the for loop
difference = np.array(fifteen_avg_list) - np.array(sixteen_avg_list)
actual_difference = fandango_2015.Fandango_Stars.mean() - fandango_2016.Fandango_Stars.mean()

Extreme_chances = np.sum(difference>abs(actual_difference))+np.sum(difference<-abs(actual_difference))
# p value
probability = Extreme_chances/20000
# As the p value is < 0.05, it means the probability that given what we observed, the chance that there is no difference is extremely unlikely --> there is a difference
# Examine the distribution of the mean differences obtained from permutation test
plt.hist(difference, bins = 30)
# add two dash lines to show where the actual difference lies in all the test samples obtained
plt.axvline(actual_difference, color = 'r', linestyle = 'dashed')
plt.axvline(-actual_difference, color = 'r', linestyle = 'dashed')
plt.show()

# bootstrap to determine 95% confidence intervals for the mean difference between 2015 and 2016 ratings
random_15 = []
random_16 = []
for i in range(10000):
	random_15.append(np.random.choice(fandango_2015.Fandango_Stars, size = fandango_2015.shape[0], replace = True).mean())
	random_16.append(np.random.choice(fandango_2016.Fandango_Stars, size = fandango_2016.shape[0], replace = True).mean())

mean_difference = np.array(random_15) - np.array(random_16)
sort_difference = np.sort(mean_difference)
# the 95% confidence intervals start from 2.5% from the left and 2.5% from the right
CI_left = sort_difference[int(0.025*10000)]
CI_right = sort_difference[int(0.975*10000)]
# this tells us that we can say the average difference in 15 and 16 ratings is 0.197 stars with a 95% confidence interval the true population mean lies between 0.08 and 0.31

