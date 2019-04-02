#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#==============================================================================
# Importing
import os, glob, re, numpy, time, random, operator
from collections import Counter
import matplotlib.pyplot as plt

#Setting random seed
random.seed(111)

# =============================================================================
# initializing global variables
# =============================================================================
list_of_dict = []
list_of_dict_with_label = []
list_updated_weights = []
list_error = []
ngram_list = []
dict_of_weights = {}

unigram = 1
bigram = 2
trigram = 3
ngram_list.append(unigram)
ngram_list.append(bigram)
ngram_list.append(trigram)


# =============================================================================
# calling create_bag_of_words function for positive and negative files path      
# =============================================================================
def call_BOW(ngram): 
    create_bag_of_words('review_polarity/txt_sentoken/pos', 1, ngram)
    create_bag_of_words('review_polarity/txt_sentoken/neg', -1, ngram)
 
# =============================================================================
# creating bag of words for each ngrams
# =============================================================================
def create_bag_of_words(file_path, label, ngram):
    file_lists = glob.glob(os.path.join(file_path , '*.txt'))
    for file_list in file_lists:
        dictionary = {}
        with open(file_list,'r') as f_input:
            text = f_input.read()
            #Adding bias term
            bias = "bias123"
            text = text + " " + bias
            #Counting the n_grams after calling generate_ngrams function
            counter = Counter(generate_ngrams(text, ngram))
            #çreating a list of dictionary for each document 
            #with counted words and a label
            dictionary['words'] = counter
            dictionary['label'] = label
            list_of_dict_with_label.append(dictionary)
            #creating a list of dictionaries for each document
            #with counted words
            list_of_dict.append(counter)
    #Removing test data from the list of counters
    for i in range(200):        
        list_of_dict.pop()        

# =============================================================================
# Function for creating n_grams
# =============================================================================
def generate_ngrams(Text, ngram):
    text = Text.lower()   
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', str(text))
    # Removing empty tokens
    words = [word for word in text.split(" ") if word != ""]  
    # Using zip method method to create ngrams 
    n_grams = zip(*[words[i:] for i in range(ngram)])
    listofngrams = [" ".join(n_gram) for n_gram in n_grams]
    return listofngrams
    
# =============================================================================
# creating a dictionary with zero weights for all n_grams
# =============================================================================
def create_weights_dictionary(list_of_dict):
    weights = {}
    #Assigning zero value to each n_gram
    for i in list_of_dict:    
        for keys, values in i.items():
            w = {keys : 0.0}
            #updating the weights dictionary
            weights.update(w)
    return weights

# =============================================================================
# function implementing perceptron algorithm to update weights
# =============================================================================
def perceptron(train_set, dict_of_weights):
    for doc_dict in train_set:
        doc_score = 0.0
        counts = doc_dict['words']
        label = doc_dict['label']
        for word, freq in counts.items():  
            #calculating the document score
            doc_score += freq * dict_of_weights[word]
        y_hat = numpy.sign(doc_score)
        #Assigning the class for zero values
        if y_hat == 0:
            y_hat = 1  
        #updating weights for wrong classifications
        if y_hat != label:
            for word, freq in counts.items():
                if label == 1:
                    dict_of_weights[word] += freq
                else:
                    dict_of_weights[word] -= freq
    #returning dictionary of updated weights            
    return dict_of_weights 

# =============================================================================
# traing model with multiple passes and shuffled train_set
# =============================================================================
def perceptron_train(train_set, test_set, dict_of_weights):
    #multiple passes
    for i in range(15):
        #shuffling the data set
        random.shuffle(train_set)
        #calling perceptron function to update weights
        dict_of_weights = perceptron(train_set, dict_of_weights)
        #checking the accuracy of classification on the test set and updated weights
        accuracy_it = test_perceptron(test_set, dict_of_weights)
        print("The accuracy for iteration ",i+1," is: ",accuracy_it,"%")
        #calculating the error percentage on the training set and updated weights
        accu_train = test_perceptron(train_set, dict_of_weights)
        list_error.append(100-accu_train)
        #creating a list of dictionaries of updated weights
        #for each iteration for averaging later
        list_updated_weights.append(dict_of_weights.copy())
    #returning the final updated dictionary after training the model 
    return dict_of_weights 

# =============================================================================
# function to average the list of dictionary of updated weights
# =============================================================================
def average_weight(new_weights):
    sums = Counter()
    counters = Counter()
    for itemset in new_weights:
        sums.update(itemset)
        counters.update(itemset.keys())
        #taking the average of each n_gram
        average_weight_dict = {x: float(sums[x])/counters[x] for x in sums.keys()}
    #returning the averaged weights dictionary
    return average_weight_dict

# =============================================================================
# testing model for perceptron algorithm
# =============================================================================
def test_perceptron(test_set, dict_of_weights):
    #c is the count of correctly classified documents
    c = 0.0
    for doc_dict in test_set:
        doc_score = 0.0
        counts = doc_dict['words']
        label = doc_dict['label']
        for word, freq in counts.items():
            #for words missing in the training set but present in test set
            if word not in dict_of_weights:
                continue
            #calculating the document score
            doc_score += freq * dict_of_weights[word]
        #incrementing c for correct classifications
        if doc_score >=0:
            if label == 1:
                c+=1
        else:
            if label == -1:
                c+=1  
    #returning the accuracy percentage
    return(c/len(test_set))*100
    
# =============================================================================
# function to plot list of accuracies to show learning progress   
# =============================================================================
def plot(error_unigram,error_bigram,error_trigram):
    plt.plot(error_unigram, marker='o', label='Unigram Error')
    plt.plot(error_bigram, marker='*', label='Bigram Error')
    plt.plot(error_trigram, marker='^', label='Trigram Error')
    plt.xlabel('Iteration-Learning Progress')
    plt.ylabel('Error Percentage on training data')
    plt.legend()
    plt.grid()
    plt.show()
    

#==============================================================================
# MAIN
    
if __name__ == '__main__':
    
    #loop to run each model of n_gram
    for item in ngram_list:
        
        #Resetting global variables for each model
        list_of_dict = []
        list_of_dict_with_label = []
        list_updated_weights = []
        dict_of_weights = {}
            
        #calling function to create bag of words with different n_grams
        call_BOW(item)
        
        print("\nRunning n_gram Model for n = ", item)
        
        #taking time stamp
        start = time.time() 
    
        #creating dictionary with zero weights
        dict_of_weights = create_weights_dictionary(list_of_dict)
        
        #Slicing data
        pos_train = list_of_dict_with_label[0:800]
        pos_test = list_of_dict_with_label[800:1000]
        neg_train = list_of_dict_with_label[1000:1800]
        neg_test = list_of_dict_with_label[1800:2000]
        
        #Creating training_set and test_set 
        train_set = pos_train + neg_train
        test_set = pos_test + neg_test
        
        #checking accuracy with zero weights
        zero_weights_accuracy = test_perceptron(train_set, dict_of_weights)
        print("\nThe accuracy with zero weights(standard binary perceptron): ", zero_weights_accuracy,"%\n")
        
        #calling perceptron algorithm and checking accuracy with randomized single pass
        random.shuffle(train_set)
        updated_weights = perceptron(train_set, dict_of_weights)
        accuracy_before_training = test_perceptron(test_set, updated_weights)
        print("The Accuracy with single pass/shuffling): ", accuracy_before_training,"%\n")
        
        #resetting zero weights
        dict_of_weights = create_weights_dictionary(list_of_dict)
        
        #calling perceptron algorithm and checking accuracy with randomized multiple passes
        updated_weights_train =perceptron_train(train_set,test_set, dict_of_weights)
        accuracy_after_training = test_perceptron(test_set, updated_weights_train)
        print("\nThe Accuracy after training model(multiple passes/shuffling): ", accuracy_after_training,"%\n")
        
        #creating average weight dictionary and checking accuracy
        average_weight_dict = average_weight(list_updated_weights)
        accuracy_on_average_weights = test_perceptron(test_set, average_weight_dict)
        print("The Accuracy with average weights(multiple passes/shuffling/averaging): ", accuracy_on_average_weights,"%\n")
        
        #printing top 10 positive weighted words
        positive_sort = sorted(updated_weights_train.items(), key=operator.itemgetter(1))[::-1]
        print("Top 10 positive weighted words: ", positive_sort[:10],"\n")
        
        #printing top 10 positive weighted words
        negative_sort = sorted(updated_weights_train.items(),key=operator.itemgetter(1))
        print("Top 10 negative weighted words: ", negative_sort[:10],"\n")
        
        #taking time stamp
        end = time.time()
        
        #printing total time taken
        print('Time taken: ',end-start,"sec\n\n\n")
     
    #slicing the list_error to plot the learning progress for each model
    error_unigram = list_error[0:15]  
    error_bigram = list_error[15:30]  
    error_trigram = list_error[30:45]  

    #plotting list of errors on each iteration(learning progress)
    plot(error_unigram,error_bigram,error_trigram)