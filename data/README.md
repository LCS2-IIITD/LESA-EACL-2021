To create splits in the Twitter dataset, use the following code:

```
dataset_twitter = pd.read_csv('Twitter.csv', encoding='utf-8')
dataset_twitter = dataset_twitter[['tweet_text', 'claim']]
dataset_twitter['claim'] = dataset_twitter['claim'].astype(float)
dataset_twitter = dataset_twitter.sample(frac=1, random_state=0).reset_index(drop=True)

## SPLIT INTO TRAIN AND TEST
split_ratio = 0.15
split_idx = int(dataset_twitter.shape[0] * (1 - split_ratio))

twitter_train = dataset_twitter.iloc[:split_idx]
twitter_test = dataset_twitter.iloc[split_idx:, :]

twitter_train.columns = ['text', 'claim']
twitter_test.columns = ['text', 'claim']

## DOWNSAMPLE TRAIN SET IN 1:1 RATIO
count_1_values, count_0_values = twitter_train['claim'].value_counts()

class_0 = twitter_train[twitter_train['claim'] == 0]
class_1 = twitter_train[twitter_train['claim'] == 1]

count_1_needed = int(count_0_values * 1)

class_1_under = class_1.sample(count_1_needed)

twitter_train = pd.concat([class_1_under, class_0], axis=0)
```