import tensorflow as tf
def cross_entropy_loss(inputs, true_w):
    """
    ==========================================================================

    inputs: The embeddings for context words. Dimension is [batch_size, embedding_size].
    true_w: The embeddings for predicting words. Dimension of true_w is [batch_size, embedding_size].

    Write the code that calculate A = log(exp({u_o}^T v_c))

    A =


    And write the code that calculate B = log(\sum{exp({u_w}^T v_c)})


    B =

    ==========================================================================
    """
    A = tf.log(tf.exp(tf.reshape(tf.reduce_sum(tf.multiply(inputs,true_w), axis = 1),[inputs.get_shape().as_list()[0],1])))

    B = tf.reshape(tf.log(tf.reduce_sum(tf.exp(tf.matmul(inputs,tf.transpose(true_w))),axis = 1)),[inputs.get_shape().as_list()[0],1])

    return tf.subtract(B, A)

def nce_loss(inputs, weights, biases, labels, sample, unigram_prob):
    """
    ==========================================================================

    inputs: Embeddings for context words. Dimension is [batch_size, embedding_size].
    weigths: Weights for nce loss. Dimension is [Vocabulary, embeeding_size].
    biases: Biases for nce loss. Dimension is [Vocabulary, 1].
    labels: Word_ids for predicting words. Dimesion is [batch_size, 1].
    samples: Word_ids for negative samples. Dimension is [num_sampled].
    unigram_prob: Unigram probability. Dimesion is [Vocabulary].

    Implement Noise Contrastive Estimation Loss Here

    ==========================================================================
    """
    ###########################################################################33
    K = len(sample)
    batch_size = inputs.get_shape().as_list()[0]
    embedding_size =  inputs.get_shape().as_list()[1]
    sample_size = len(sample)
    delta = tf.exp(-10.0)

    # Lookup for fetching the embeddings for the labels
    label_embedding = tf.reshape(tf.nn.embedding_lookup(weights,labels,name="labels_embedding"),[batch_size,embedding_size])
    
    # Lookup for fetching the embeddings for the samples
    sample_embedding = tf.reshape(tf.nn.embedding_lookup(weights,sample,name="sample_embedding"),[sample_size,embedding_size])

    # Lookup for fetching the bias for the samples
    sample_bias = tf.reshape(tf.nn.embedding_lookup(biases,sample,name="sample_bias"),[sample_size,1])
    
    
    unigram_prob = tf.reshape(unigram_prob,[weights.get_shape().as_list()[0],1])
    
    # Lookup for fetching the unigram probabilities for the sample
    sample_prob = tf.reshape(tf.nn.embedding_lookup(unigram_prob, sample, name = "unigram_sample"),[sample_size,1])
    
    # Matrix multiplication for samples and inputs {sample*batch_size}
    sample_matmul = tf.matmul(sample_embedding,inputs,transpose_b = True)
    
    # Replicating the sample bias for easy addition
    sample_bias_multiple = tf.tile(sample_bias, [1,batch_size])
    
    s_wxwc = tf.add(sample_matmul , sample_bias_multiple)
    
    # Replicating the probabilities for samples for easy arithematic
    sample_prob_multiple = tf.tile(sample_prob, [1,batch_size])
    
    k_sample_prob_multiple = tf.scalar_mul(K, sample_prob_multiple)
    
    log_k_sample = tf.log(k_sample_prob_multiple + delta)
    
    sub_swxwc_logk_sample = tf.subtract(s_wxwc,log_k_sample, name = "Inner-sigmoid-B")
    
    sigmoid_wxwc = tf.sigmoid(sub_swxwc_logk_sample,name = "sigmoid-B")

    log_red_sum_sample = tf.log(1-sigmoid_wxwc+delta)

    red_sum_sample = tf.reduce_sum(log_red_sum_sample,[0])
    #######################################################################################
    # Lookup for fetching the biases for the labels
    label_bias = tf.reshape(tf.nn.embedding_lookup(biases,labels,name="label_bias"),[batch_size,1])
    
    # Lookup for fetching the unigram probabilities for the labels
    label_prob = tf.reshape(tf.nn.embedding_lookup(unigram_prob, labels, name = "unigram_sample"),[batch_size,1])
    
    # Matrix multiplication and taking the diagonal elements
    label_matmul = tf.reshape(tf.diag_part(tf.matmul(label_embedding,inputs,transpose_b = True)),[batch_size,1])

    s_wowc = tf.add(label_matmul , label_bias)
  
    k_label_prob_multiple = tf.scalar_mul(K, label_prob)
  
    log_k_label = tf.log(k_label_prob_multiple)  
   
    sub_swowc_logk_label = tf.subtract(s_wowc,log_k_label, name = "Inner-sigmoid-B")
   
    sigmoid_wowc = tf.sigmoid(sub_swowc_logk_label,name = "sigmoid-B")

    log_red_sum_label = tf.log(sigmoid_wowc + delta)

    final_sum = tf.add(red_sum_sample,log_red_sum_label)
    
    return tf.negative(final_sum)