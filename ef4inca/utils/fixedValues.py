#### Normalization values of the dataset! Computed from the training set, no 2023 data is used for this. ###
NORM_DICT_TOTAL = {
    'ch01': {'mean': 10.30721493, 'std': 17.11751967},
    'ch02': {'mean': 12.52159241, 'std': 18.74300191},
    'ch03': {'mean': 8.3304175, 'std': 11.42169462},
    'ch04': {'mean': 272.52816293, 'std': 29.05951619},
    'ch05': {'mean': 230.35037452, 'std': 11.17119498}, 
    'ch06': {'mean': 243.4373616, 'std': 17.55141617}, 
    'ch07': {'mean': 262.8061038, 'std': 28.81357914},
    'ch08': {'mean': 252.71339021, 'std': 19.3080578},
    'ch09': {'mean': 262.36661109, 'std': 29.86794278},
    'ch10': {'mean': 259.70074305, 'std': 29.29611851},
    'ch11': {'mean': 246.91890268, 'std': 21.28191747},
    'dem': {'mean': 183.85984719055546, 'std': 123.33213630309506},
    'lat': {'mean': 7.900, 'std': 1.96297496}, 
    'lon': {'mean': -1.175, 'std': 1.4577879},
    }
# This is from the repository of Caglar, not used in this code
#### Hand-picked events for best and worst performance from the test dataset
bestSamples = ['sampled_202305051455', 'sampled_202307130340', 'sampled_202307050335',
               'sampled_202308261550', 'sampled_202307041510']

worstSamples = ['sampled_202307111910', 'sampled_202307122015', 'sampled_202306271725',
               'sampled_202306231615', 'sampled_202308051200']

#### Randomly selected events from the remaining of the test dataset
randSamples = ['sampled_202308061645', 'sampled_202307211455', 'sampled_202307292200',
               'sampled_202307130125', 'sampled_202307151635', 'sampled_202307250800',
               'sampled_202308291420', 'sampled_202308021355', 'sampled_202308171200', 'sampled_202307040235']