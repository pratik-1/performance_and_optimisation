## 1. Index Optimalization: Index as much as possible for merging and value lookup

# Merge
%%timeit
listings.merge(reviews, on='listing_id')
# 439 ms ± 24.5 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

%%timeit
reviews_ = reviews.set_index('listing_id')
listings_ = listings.set_index('listing_id')
listings_.merge(reviews_, left_index=True, right_index=True)
# 393 ms ± 17.4 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

# Search

listings = listings.set_index('listing_id', drop=False)

%%timeit
listings.loc[29844866, 'name']
# 10.1 µs ± 1.25 µs per loop (mean ± std. dev. of 7 runs, 100000 loops each)

%%timeit
listings.at[29844866, 'name']
# 5.34 µs ± 474 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)

listings = listings.reset_index(drop=True)

%%timeit
listings.loc[listings['listing_id'] == 29844866, 'name']
# 593 µs ± 30 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)

%%timeit 
listings.iloc[22529]['name']
# 252 µs ± 45.9 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)



## 2. Vectorize Operations: The process of executing operations on entire arrays

# .iloc[]
%%timeit
norm_prices = np.zeros(len(listings,))
for i in range(len(listings)):
    norm_prices[i] = (listings.iloc[i]['price'] - min_price) / (max_price - min_price)
listings['norm_price'] = norm_prices
# 8.91 s ± 479 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

# Iterrows()
%%timeit
norm_prices = np.zeros(len(listings,))
for i, row in listings.iterrows():
    norm_prices[i] = (row['price'] - min_price) / (max_price - min_price)
listings['norm_price'] = norm_prices
# 3.99 s ± 346 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

# .loc[]
%%timeit
norm_prices = np.zeros(len(listings,))
for i in range(len(norm_prices)):
    norm_prices[i] = (listings.loc[i, 'price'] - min_price) / (max_price - min_price)
listings['norm_price'] = norm_prices
# 408 ms ± 61.2 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

# .map()
%%timeit 
listings['norm_price'] = listings['price'].map(lambda x: (x - min_price) / (max_price - min_price))
# 39.8 ms ± 2.33 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)

# Vectorize
%%timeit
listings['norm_price'] = (listings['price'] - min_price) / (max_price - min_price)
# 1.76 ms ± 107 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)



# When logic is complex for vectorisation

room_type_scores = {'Entire home/apt': 1,
                   'Private room': 0.7,
                   'Shared room': 0.2}
# iloc
%%timeit
scores = np.zeros(len(listings))
for i in range(len(listings)):
    row = listings.loc[i]
    if row['availability_365'] == 0:
        scores[i] = 0
    elif row['price'] > 100:
        scores[i] = 0
    else:
        room_type_score = room_type_scores[row['room_type']]
        price_score = (100 - row['price']) / 100
        review_score = 1 if row['number_of_reviews'] > 50 else 0.5
        scores[i] = room_type_score * price_score * review_score
listings['score'] = scores
# 5.64 s ± 194 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)


# Numpy: Taking data out when vectorising on complex logic
# Try to avoid looping, using native numpy, or .map()

%%timeit
prices = listings['price'].values
nr_reviews = listings['number_of_reviews'].values
availability = listings['availability_365'].values
room_types = listings['room_type'].values
scores = np.zeros(len(listings))
for i in range(len(listings)):
    if availability[i] == 0:
        scores[i] = 0
    elif prices[i] > 100:
        scores[i] = 0
    else:
        room_type_score = room_type_scores[room_types[i]]
        price_score = (100 - prices[i]) / 100
        review_score = 1 if nr_reviews[i] > 50 else 0.5
        scores[i] = room_type_score * price_score * review_score
listings['score'] = scores
# 41.4 ms ± 2.31 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)

# Vectorize: Best optimisation

%%timeit 
listings.loc[listings['room_type'] == 'Entire home/apt', 'room_type_score'] = 1
listings.loc[listings['room_type'] == 'Private room', 'room_type_score'] = 0.7
listings['room_type_score'].fillna(0.2, inplace=True)
listings.loc[listings['number_of_reviews'] > 50, 'review_score'] = 1
listings['review_score'].fillna(0.5, inplace=True)
listings['price_score'] = (100 - listings['price']) / 100
listings['score'] = listings['room_type_score'] * listings['price_score'] * listings['review_score']
listings.loc[(listings['availability_365'] == 0) | 
             (listings['price'] > 100), 'score'] = 0
# 17.5 ms ± 668 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)



## 3. Memory Optimization: When reading in a csv or json file the column types are inferred and are defaulted to the largest data type (int64, float64, object).

# Use functions that will downcast the columns automatically to the smallest possible datatype
# Use pandas category column type for categorical string
# Date columns we cast to the pandas datetime dtype. It does not reduce memory usage, but enables time based operations.


def optimize_floats(df: pd.DataFrame) -> pd.DataFrame:
    floats = df.select_dtypes(include=['float64']).columns.tolist()
    df[floats] = df[floats].apply(pd.to_numeric, downcast='float')
    return df

def optimize_ints(df: pd.DataFrame) -> pd.DataFrame:
    ints = df.select_dtypes(include=['int64']).columns.tolist()
    df[ints] = df[ints].apply(pd.to_numeric, downcast='integer')
    return df

def optimize_objects(df: pd.DataFrame, datetime_features: List[str]) -> pd.DataFrame:
    for col in df.select_dtypes(include=['object']):
        if col not in datetime_features:
            if not (type(df[col][0])==list):
                num_unique_values = len(df[col].unique())
                num_total_values = len(df[col])
                if float(num_unique_values) / num_total_values < 0.5:
                    df[col] = df[col].astype('category')
        else:
            df[col] = pd.to_datetime(df[col])
    return df

def optimize(df: pd.DataFrame, datetime_features: List[str] = []):
    return optimize_floats(optimize_ints(optimize_objects(df, datetime_features)))


# 4. Filter Optimization
# When chaining multiple operations, filter data in early stages.
