#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from numpy.random import choice
import numpy as np
from collections import Counter
import scipy.stats as stats
from IPython.display import display, HTML
from statsmodels.stats.inter_rater import fleiss_kappa
from statistics import mode
from tqdm import tqdm_notebook as tqdm
import random


# In[2]:


class dataset:
    def __init__(self, data, label_name, name):
        self.data = data
        #The column name for the label in the dataset
        self.label_name = label_name
        #name of dataset
        self.name = name


# In[3]:


#Trinarize data to be consistent to 2 or 3 categories
def trinarize_misinfo(input):
    if (input == "Very high credibility" or input == "Somewhat high credibility"):
        return 1
    elif (input == "Medium credibility"):
        return 0
    else:
        return -1


# In[4]:


def trinarize_snli(input):
    if(input == 'entailment'):
        return 1
    elif(input == 'neutral'):
        return 0
    else:
        return -1


# In[5]:


def trinarize_sentiment(input):
    if(int(input) > 0):
        return 1
    elif(int(input) == 0):
        return 0
    else:
        return -1


# In[6]:


#Reading data and grouping annotations by tasks
def get_data(data_file):
    if (data_file == 'toxicity'):
        data = pd.read_csv("data/TOXICITY_toxicity_individual_annotations.csv")
        grouped = data.groupby("id")[['id', 'toxic']]
    
    if (data_file == 'misinfo'):
        data = pd.read_csv("data/MISINFO_misinfo_credco_2019_study_cplusj_2020_subset.csv")
        annotators = pd.read_csv("data/MISINFO_misinfo_credco_study_2019_crowd_annotators_simple.csv")
        
        #combine data with annotators
        data = data[data['annotator'].isin(annotators[annotators['pool']=='Upwork']['annotator_id'].values)]
        
        #one annotator that had 26 labels insteaad of 25
        data = data[data['annotator'] != 'CredCo-3AA.33']
        data = data[['report_title', 'task_1_answer']]
        data['task_1_answer'] = [trinarize_misinfo(x) for x in data['task_1_answer']]
        grouped = data.groupby("report_title")
    
    if (data_file == 'pg13'):
        data = pd.read_csv("data/PG13_labels.txt", sep='\t', header=None)
        #keep only annotations with at least ten labels
        filtered = data.groupby(1)[1].filter(lambda x: len(x) >= 10)
        grouped = data[data[1].isin(filtered)].groupby(1)
    
    if (data_file == 'bots'):
        data = pd.read_csv("data/BOTS_crowdflower_results_detailed.csv")[['crowdflower_id', 'class']]
        #keep only annotations with at least three labels
        filtered = data.groupby('crowdflower_id')['crowdflower_id'].filter(lambda x: len(x) >= 3)
        grouped = data[data['crowdflower_id'].isin(filtered)].groupby('crowdflower_id')

    if (data_file == 'snli'):
        # snli check
        # dynamic worker routing
        data = pd.read_json("data/SNLI_snli_1.0_train.jsonl", lines=True)
        labels = []
        for name, row in data.iterrows():
        #keep only annotations with at least five labels
            if len(row["annotator_labels"]) >= 5:
                for label in row["annotator_labels"]:
                    labels.append({"idx": name, "label": label})
        data = pd.DataFrame(labels)
        data['label'] = [trinarize_snli(x) for x in data[['label']].values]
        data = data[['idx', 'label']]
        grouped = data.groupby("idx")
        
    if (data_file == 'sentiment'):
        data = pd.read_table("data/SENTIMENT_cred_event_TurkRatings.data", sep='\t')
        labels = []
        for name, row in data.iterrows():
            #keep only annotations with at least thirty labels
            if len(row["Cred_Ratings"]) >= 30:
                row = row['Cred_Ratings'][1:-1].split(",")
                for label in row:
                    labels.append({"idx": name, "label": label})
        data = pd.DataFrame(labels)
        data['label'] = [trinarize_sentiment(int(x[0].strip("\', \""))) for x in data[['label']].values]
        data = data[['idx', 'label']]
        grouped = data.groupby("idx")
    
    if (data_file == 'wsd'):
        data = pd.read_csv("data/WSD_wsd.standardized.tsv", sep='\t')
        grouped = data.groupby("orig_id")
        
    return grouped


# In[7]:


#find a way to generate samples and finding the majority from each sample
def find_majority(values, sample_size):
    random = choice(values, sample_size * 2, replace=True)
    sample1 = Counter(random[:sample_size]).most_common(sample_size)
    sample2 = Counter(random[sample_size:]).most_common(sample_size)
    majority_1 = sample1[0][0]
    majority_2 = sample2[0][0]  
    if ((len(sample1) > 1 and sample1[0][1] == sample1[1][1])): #no majority
        majority_1 = 'NM'
    if (len(sample2) > 1 and sample2[0][1] == sample2[1][1]): #no majority
        majority_2 = 'NM'
    return sample1, sample2, majority_1, majority_2


# In[8]:


#Create parellel datasets of size 1, 3, 5, 7, 9
#and compare flip rate between the parallel datasets
#Resample 100 times and take the average of these iterations
def resample_flipping(data_object):
    for sample_size in [1, 3, 5, 7, 9]:
        flip_count = 0
        for iterations in tqdm(range(0, 100)):
            for name, group in data_object.data:
                sample1, sample2, majority_1, majority_2 = find_majority(group[data_object.label_name].values, sample_size)
                if (majority_1 != majority_2):
                    flip_count += 1

        print('sample size', sample_size)
        print("flip average percent", round(flip_count/len(data_object.data), 2))


# In[9]:


#Create a parallel dataset, how many of these labels flip?
#Create another parallel dataset, how many of original labels flipped at least once?
#Repeat this process for up to 5 parallel datasets
#Repeat this process 100 times and take the average number of flips for 1, 2, 3, 4, 5 parallele datasets
def population_level(data_object):
    sample_size = 5
    percent_map = {}
    for x in tqdm(range(0, 100)):
        flipped_labels = {}
        label_count = {}

        for iteration in range(1, 6):
            temp = []
            for name, group in data_object.data:
                values = group[data_object.label_name].values
                #find the gold label for the data point
                actual_label = Counter(values).most_common(len(values))[0][0]
                #keep track of how many gold labels there are

                if (actual_label not in label_count):
                    label_count[actual_label] = set()
                    flipped_labels[actual_label] = set()
                
                label_count[actual_label].add(name) 
                #keep track of how many labels in this category flipped
                sample1, sample2, majority_1, majority_2 = find_majority(group[data_object.label_name].values, sample_size)
                if (majority_1 != majority_2):
                    flipped_labels[actual_label].add(name)

            for actual_label in label_count:
                percent = round(len(flipped_labels[actual_label])/len(label_count[actual_label]), 2)
                key = "iteration: " + str(iteration) + " label:" + str(actual_label)
                percent_map[key] = round(percent_map.get(key, 0) + percent, 2)
    for x in percent_map:
        print(x, percent_map[x])

    


# In[10]:


#Preprocess data for each of the data to calculate
#fleiss kappa scores
def calculate_fleiss_kappa(data_object):
    result = []
    if (data_object.name == 'pg13'):        
        map = {'G':0, 'P':1, 'X':2, 'R':3}
        for i, x in data_object.data:
            #need at least five samples
            #create a sample out of 5
            if (len(list(x[data_object.label_name].values)) >= 5):
                counts = Counter(random.sample(list(x[data_object.label_name].values), 5))
                temp = [0] * 4
                for y in counts:
                    temp[map[y]] = counts[y]
                result.append(temp)
            
    if (data_object.name == 'misinfo' or data_object.name == 'snli'):        
        for i, x in data_object.data:
            counts = Counter(x[data_object.label_name].values)
            temp = [0] * 3
            for y in counts:
                temp[y + 1] = counts[y]
            result.append(temp)
    
    if (data_object.name == 'wsd'):        
        for i, x in data_object.data:
            counts = Counter(x[data_object.label_name].values)
            temp = [0] * 3
            for y in counts:
                temp[y - 1] = counts[y]
            result.append(temp)
    
    if (data_object.name == 'sentiment'):
        for i, x in data_object.data:
            if (len(list(x[data_object.label_name].values)) >= 30):
                counts = Counter(x[data_object.label_name].values)
                temp = [0] * 3
                for y in counts:
                    temp[y + 1] = counts[y]
                result.append(temp)
    
    if (data_object.name =='toxic'):
        for name, group in tqdm(data_object.data):
            subsample = group.sample(5, replace = True)
            temp = subsample['toxic'].values
            result.append(temp)

        result = pd.DataFrame(result)
        result['toxic'] = result.sum(axis=1)
        result['non_toxic'] = 5 - result['toxic']
        result = result[['toxic', 'non_toxic']].values
    
    if (data_object.name == 'bots'):        
        map = {'genuine':0, 'spambot':1, 'unable':2}
        for i, x in data_object.data:
            if (len(list(x[data_object.label_name].values)) >= 3):
                counts = Counter(x[data_object.label_name].values)
                temp = [0] * 3
                for y in counts:
                    temp[map[y]] = counts[y]
                result.append(temp)
    print("fleiss kappa score: ", fleiss_kappa(result, method='fleiss'))


# In[11]:


#create parallel datasets
def calculate_class_difference_helper(data_object):
    sample_size_to_resampled_dataset = {}

    for sample_size in [5, 7, 9]:
        resampled_datasets = [[], [],[], []]
        for iterations in tqdm(range(0, 100)):
            for name, group in data_object.data:
                sample1, sample2, majority_1, majority_2 = find_majority(group[data_object.label_name].values, sample_size)

                resampled_datasets[0].append(name)
                resampled_datasets[1].append(majority_1)
                resampled_datasets[2].append(majority_2)
                to_append = 0 # assume no difference

                if (majority_1 != majority_2):
                    # in case sample2 didn't have any votes for the majority class from sample1
                    to_append= abs(sample1[0][1])
                    for x in sample2:
                        if (x[0] == majority_1):
                            to_append = abs(sample1[0][1] - x[1])
                resampled_datasets[3].append(to_append)
        sample_size_to_resampled_dataset[sample_size] = resampled_datasets
    return sample_size_to_resampled_dataset


# In[12]:


#across the parallel datasets, calculate how large of a difference exists
#between contested labels
def calculate_difference_of_flip(sample_size_to_resampled_dataset):
    for sample_size in sample_size_to_resampled_dataset:
        print("sample size", sample_size)
        resampled_dataset = sample_size_to_resampled_dataset[sample_size]
        results = pd.DataFrame(resampled_dataset).transpose()
        
        #get rid of no modes
        results = results[results[1] != 'NM']
        results = results[results[2] != 'NM']
        results[3] = pd.to_numeric(results[3])
        
        print("mean")
        print(round(results[results[1] != results[2]][[3]].mean().values[0], 2))
        print("std")
        print(round(results[results[1] != results[2]][[3]].std().values[0], 2))


# In[13]:


#Across the parallel datasets, calculate the flip rate for each class
#given sample size of 5
def calculate_flips_per_class(sample_size_to_resampled_dataset):
    resampled_dataset = sample_size_to_resampled_dataset[5]
    resampled_dataset = pd.DataFrame(resampled_dataset).transpose()
    
    grouped = resampled_dataset[resampled_dataset[1] != resampled_dataset[2]].groupby([1]).count()
#     display(grouped)
    counts = resampled_dataset.groupby([1]).count()
#     display(counts)
    print(round(grouped[0]/counts[0],2))


# In[14]:


#run any of the tests
def run_test(datasets, test, helper=False):
    for dataset in datasets:
        print('------------------------')
        print(dataset.name)
        if (helper):
            sample_size_to_resampled_dataset = calculate_class_difference_helper(dataset)
            calculate_difference_of_flip(sample_size_to_resampled_dataset)
            calculate_flips_per_class(sample_size_to_resampled_dataset)
        else:
            test(dataset)
        print()


# In[15]:


#Create all the data objects
toxicity_object = dataset(get_data("toxicity"), 'toxic', 'toxic')
misinfo_object = dataset(get_data("misinfo"), 'task_1_answer', 'misinfo')
pg13_object = dataset(get_data("pg13"), 2, 'pg13')
bots_object =  dataset(get_data("bots"), 'class', 'bots')
snli_object = dataset(get_data("snli"), 'label', 'snli')
sentiment_object = dataset(get_data("sentiment"), 'label', 'sentiment')
wsd_object = dataset(get_data("wsd"), 'response', 'wsd')


# In[ ]:


list_of_datasets = [toxicity_object, misinfo_object, pg13_object, bots_object, snli_object, sentiment_object, wsd_object]
#, 
#test 1 - Resample Flip
run_test(list_of_datasets, resample_flipping)
# #test 2 - Population Level
# run_test(list_of_datasets, population_level)
# #test 3 - Fleiss Kappa
# run_test(list_of_datasets, calculate_fleiss_kappa)
# Helper + test 4, test 5 - Difference in Flip, Flips across Classes
# run_test(list_of_datasets, None, True)


# In[ ]:





# In[ ]:




