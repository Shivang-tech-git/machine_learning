
# Project 1 - Evaluating similarity based on correlation
import numpy as np
import pandas as pd

frame =  pd.read_csv('rating_final.csv')
cuisine = pd.read_csv('chefmozcuisine.csv')
geodata = pd.read_csv('geoplaces2.csv', encoding = 'mbcs')
frame.head()
geodata.head()
#@title
places =  geodata[['placeID', 'name']]
places.head()
cuisine.head()
## Grouping and Ranking Data
rating = pd.DataFrame(frame.groupby('placeID')['rating'].mean())
rating.head()
rating['rating_count'] = pd.DataFrame(frame.groupby('placeID')['rating'].count())
rating.head()
rating.describe()
rating.sort_values('rating_count', ascending=False).head()
print(places[places['placeID']==135085])
print(cuisine[cuisine['placeID']==135085])
## Preparing Data For Analysis
places_crosstab = pd.pivot_table(data=frame, values='rating', index='userID', columns='placeID')
places_crosstab.head()
Tortas_ratings = places_crosstab[135085]
print(Tortas_ratings[Tortas_ratings>=0])
## Evaluating Similarity Based on Correlation
similar_to_Tortas = places_crosstab.corrwith(Tortas_ratings)
corr_Tortas = pd.DataFrame(similar_to_Tortas, columns=['PearsonR'])
corr_Tortas.dropna(inplace=True)
print(corr_Tortas.head())
Tortas_corr_summary = corr_Tortas.join(rating['rating_count'])
print(Tortas_corr_summary[Tortas_corr_summary['rating_count']>=10].sort_values('PearsonR', ascending=False).head(10))
places_corr_Tortas = pd.DataFrame([135085, 132754, 135045, 135062, 135028, 135042, 135046], index = np.arange(7), columns=['placeID'])
summary = pd.merge(places_corr_Tortas, cuisine,on='placeID')
print(summary)
places[places['placeID']==135046]
cuisine['Rcuisine'].describe()


## Project 2 - Classification based collaborative filtering

from sklearn.linear_model import LogisticRegression

bank_full = pd.read_csv('bank_full_w_dummy_vars.csv')
bank_full.head()
bank_full.info()
X = bank_full.ix[:,(18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36)].values
y = bank_full.ix[:,17].values
LogReg = LogisticRegression()
LogReg.fit(X, y)
new_user = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
y_pred = LogReg.predict(new_user)
y_pred

##  Project 3 - Model based collaborative filtering systems

from sklearn.decomposition import TruncatedSVD
### Preparing the data
columns = ['user_id', 'item_id', 'rating', 'timestamp']
frame = pd.read_csv('ml-100k/u.data', sep='\t', names=columns)
frame.head()
columns = ['item_id', 'movie title', 'release date', 'video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure',
          'Animation', 'Childrens', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
          'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
movies = pd.read_csv('ml-100k/u.item', sep='|', names=columns, encoding='latin-1')
movie_names = movies[['item_id', 'movie title']]
movie_names.head()
combined_movies_data = pd.merge(frame, movie_names, on='item_id')
combined_movies_data.head()
combined_movies_data.groupby('item_id')['rating'].count().sort_values(ascending=False).head()
filter = combined_movies_data['item_id']==50
combined_movies_data[filter]['movie title'].unique()
### Building a Utility Matrix
rating_crosstab = combined_movies_data.pivot_table(values='rating', index='user_id', columns='movie title', fill_value=0)
rating_crosstab.head()
### Transposing the Matrix
rating_crosstab.shape
X = rating_crosstab.T
X.shape
### Decomposing the Matrix
SVD = TruncatedSVD(n_components=12, random_state=17)
resultant_matrix = SVD.fit_transform(X)
resultant_matrix.shape
### Generating a Correlation Matrix
corr_mat = np.corrcoef(resultant_matrix)
corr_mat.shape
### Isolating Star Wars From the Correlation Matrix
movie_names = rating_crosstab.columns
movies_list = list(movie_names)
star_wars = movies_list.index('Star Wars (1977)')
star_wars
corr_star_wars = corr_mat[1398]
corr_star_wars.shape
### Recommending a Highly Correlated Movie
list(movie_names[(corr_star_wars<1.0) & (corr_star_wars > 0.9)])
list(movie_names[(corr_star_wars<1.0) & (corr_star_wars > 0.95)])


## Project 4 - Content based recommender systems

from sklearn.neighbors import NearestNeighbors

cars = pd.read_csv('mtcars.csv')
cars.columns = ['car_names', 'mpg', 'cyl', 'disp', 'hp', 'drat', 'wt', 'qsec', 'vs', 'am', 'gear', 'carb']
cars.head()
t = [15, 300, 160, 3.2]
X = cars.ix[:,(1, 3, 4, 6)].values
X[0:5]
nbrs = NearestNeighbors(n_neighbors=1).fit(X)
print(nbrs.kneighbors([t]))
cars


