# Data retrived from https://www.dataquest.io/blog/data-science-portfolio-project-finding-the-two-best-markets-to-advertise-in-an-e-learning-product/
# I looked at variables I am interested in in this data set 

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
direct_link = 'https://raw.githubusercontent.com/freeCodeCamp/2017-new-coder-survey/master/clean-data/2017-fCC-New-Coders-Survey-Data.csv'
fcc = pd.read_csv(direct_link, low_memory = 0) 
pd.options.display.max_columns = 150

# Get an overview on the coders' general job interest
fcc.JobRoleInterest.value_counts(dropna = False, normalize = True)
# 61.5% of the data are not available. This could mean most coders did not know for certain what they would do.
# The second highest choice is full-stack web developer
# Because majority of the JobRoleInterest is not available, to make examination of the data easier, new data frame is generated to analyze when JobRoleInterest is available
JobRoleInterest_no_NaN = fcc.JobRoleInterest.dropna()
split_interests = JobRoleInterest_no_NaN.str.split(',')
person_choices = split_interests.apply(lambda x:len(x))
person_choices.value_counts(normalize = True).sort_index()
# 31.65% of the participants had 1 job interest in mind while 11% had 2 and the rest had more
# Make new dataframe to see the number of coders in each country
Country_df = fcc[['JobRoleInterest','CountryLive']].copy()
country_count = Country_df['CountryLive'].value_counts()
country_count_perc = Country_df['CountryLive'].value_counts(normalize = True)*100
# Top 5 countries with the most coders
country_top5 = country_count[:5,]
# Shorten the country names so easier to read
country_top5.index = ['US', 'India', 'UK', 'Canada', 'Brazil']
country_top5_perc = country_count_perc[:5,]
country_top5_perc.index = ['US', 'India', 'UK', 'Canada', 'Brazil']
# make bar graphs using seaborn
f,axes = plt.subplots(1,2, figsize=(12,5))
sns.barplot(country_top5.index, country_top5.values, alpha = 0.8, ax = axes[0])
sns.barplot(country_top5_perc.index, country_top5_perc.values, alpha = 0.8, ax = axes[1])
axes[0].set_title('Coder number by countries')
axes[1].set_title('Coder % by countries')
axes[0].set(ylabel = 'number of coders')
axes[1].set(ylabel = '% of coders')
plt.rcParams["axes.labelsize"] = 15

plt.show()
# This tells us that the top 5 countries with new coders are US, India, UK, Canada and Brzil
US_df = Country_df[Country_df.CountryLive == 'United States of America'].copy()
US_df.JobRoleInterest.value_counts(dropna= False,normalize = True)
# This tells us that in US, majority of the new coders don't know what they will do and the second highest choice is full-stack web developer. 
# The data made sense since US has the highest new coder population in the entire dataset so the trend is similar to the entire data set. 

# I am curious to know the gender ratio of the new coders in the top 3 countries with the most coders
Country_df2 = fcc[['CountryLive','Gender']]
US_count = Country_df2[Country_df2.CountryLive == 'United States of America'].Gender.value_counts(normalize = True)*100
India_count = Country_df2[Country_df2.CountryLive == 'India'].Gender.value_counts(normalize = True)*100
UK_count = Country_df2[Country_df2.CountryLive == 'United Kingdom'].Gender.value_counts(normalize = True)*100

combined_df = pd.DataFrame(data = {'US':US_count, 'India':India_count, 'UK':UK_count })
# plot bar graphs grouped by countries
barWidth = 0.2
x1 = np.arange(5)
x2 = [x + barWidth for x in x1]
x3 = [x + barWidth for x in x2]
plt.bar(x1, combined_df.US, width = barWidth, label = 'US')
plt.bar(x2, combined_df.India, width = barWidth, label = 'India')
plt.bar(x3, combined_df.UK, width = barWidth, label = 'UK')
plt.xticks(np.arange(5), combined_df.index)
plt.ylabel('Percentage')
plt.title('Coder distribution by gender by countries')
plt.legend()

# I am also curious to know what coders generally major in
fcc.SchoolMajor.value_counts(normalize = True)

biology = fcc.SchoolMajor.str.contains('Biology')
biology.value_counts(normalize = True)
# Turned out that only 2% of the coders major in biology like me. 
drop_na = fcc.SchoolMajor.dropna()
biology2 = drop_na[drop_na.str.contains('Biology')]
biology2.value_counts(normalize = True)
# Of the 2% biologists, only 8.5% major in biochemistry and molecular biology like me so that is 0.17% of the total coders. I am a rare species.
# Last, I can find out if the hours spent on learning differs by major
groupby_major = fcc.groupby('SchoolMajor')
groupby_major.HoursLearning.mean().sort_values(ascending = False)
