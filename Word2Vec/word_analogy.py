import os
import pickle
import numpy as np
# Added scipy to access cosine similarity
from scipy import spatial

model_path = './models/'
loss_model = 'cross_entropy'
#loss_model = 'nce'

model_filepath = os.path.join(model_path, 'word2vec_%s.model'%(loss_model))

dictionary, steps, embeddings = pickle.load(open(model_filepath, 'r'))
"""
==========================================================================

Write code to evaluate a relation between pairs of words.
You can access your trained model via dictionary and embeddings.
dictionary[word] will give you word_id
and embeddings[word_id] will return the embedding for that word.

word_id = dictionary[word]
v1 = embeddings[word_id]

or simply

v1 = embeddings[dictionary[word_id]]

==========================================================================
"""
examples_file  = open("word_analogy_test.txt", "r")
result_file = open("word_analogy_test_predictions_cross_entropy.txt","w")     # Writing the results to a file

for line in examples_file:
    cos_sim_pairs_diff = []
    line.strip()
    examples = line.strip().split("||")[0]
    example_pairs = examples.strip().split(",")
    avg_cos_sim_examples = 0.0
    # Calculating the difference vectors for each examples
    # Then taking the average of these differences
    for k in range(0, len(example_pairs)):
        first_word = example_pairs[k].strip().split(":")[0].replace('"','')
        second_word = example_pairs[k].strip().split(":")[1].replace('"','')
        first_word_embedding = embeddings[dictionary[first_word]]
        second_word_embedding = embeddings[dictionary[second_word]]

        result_embedding = first_word_embedding - second_word_embedding
        avg_cos_sim_examples = avg_cos_sim_examples + result_embedding

    avg_cos_sim_examples = (avg_cos_sim_examples)/3
    # Calculating the difference vector for each choice
    # Then taking cosine similarity between each difference vector of choice
    # with the average difference vector we got from the examples
    # and storing it in a list
    choices = line.strip().split("||")[1]
    pairs = choices.strip().split(",")
    for j in range(0,len(pairs)):
        first_word = pairs[j].strip().split(":")[0].replace('"','')
        second_word = pairs[j].strip().split(":")[1].replace('"','')
        first_word_embedding = embeddings[dictionary[first_word]]
        second_word_embedding = embeddings[dictionary[second_word]]
        diff_embedding = first_word_embedding - second_word_embedding
        cos_diff = 1- spatial.distance.cosine(diff_embedding, avg_cos_sim_examples)

        cos_sim_pairs_diff.append(cos_diff)
        result_file.write(pairs[j]+" ")
  
    # Fetching the maximum and minimum difference 
    min_index = np.argmin(cos_sim_pairs_diff)
    result_file.write(pairs[min_index]+" ")
    max_index = np.argmax(cos_sim_pairs_diff)
    result_file.write(pairs[max_index]+"\n")

result_file.close()
