import sys, math
import numpy as np


def parse_train_tsv(filename):
    words = []
    labels = []
    words_unique = []
    labels_unique = []
    tsv_file = open(filename, "r")
    for line in tsv_file.readlines():
        if line == "\n":
            continue

        elements = line.rstrip().split("\t")
        word = elements[0]
        label = elements[1]

        words.append(word)
        if word not in words_unique:
            words_unique.append(word)

        labels.append(label)
        if label not in labels_unique:
            labels_unique.append(label)

    labels_unique.sort()
    return words, labels, words_unique, labels_unique


def parse_tsv(filename):
    words = []
    labels = []
    tsv_file = open(filename, "r")
    for line in tsv_file.readlines():
        if line == "\n":
            continue

        elements = line.rstrip().split("\t")
        word = elements[0]
        label = elements[1]

        words.append(word)
        labels.append(label)
    return words, labels


def construct_feature_vector(words, words_unique):
    feature_vector = []
    for word in words:
        m = words_unique.index(word)
        feature_vector.append(m)
    return feature_vector


def calculate_probability(epoch, feature_vector, labels, labels_unique, words_unique):
    theta = np.zeros(shape=(len(words_unique) + 1, len(labels_unique)))
    for z in xrange(0, epoch):
        for i in xrange(0, len(labels)):
            temp_theta = np.zeros(shape=(len(words_unique) + 1, len(labels_unique)))
            for k in xrange(0, len(labels_unique)):
                j_theta = calculate_gradient(i, k, theta, feature_vector, labels, labels_unique, words_unique)
                temp_theta[:, k] = 0.5 * j_theta
            theta = theta - temp_theta
    return theta


def calculate_gradient(i, k, theta, feature_vector, labels, labels_unique, words_unique):

    j_theta = np.zeros(shape=(len(words_unique) + 1))
    if labels[i] == labels_unique[k]:
        indicator = 1
    else:
        indicator = 0

    m, bias = feature_vector[i], len(words_unique)
    theta_k_t = theta[:, k]. transpose()
    theta_k_t_x_i = theta_k_t[m] + theta_k_t[bias]
    numerator = math.exp(theta_k_t_x_i)
    denominator = 0
    for j in xrange(0, len(labels_unique)):
        theta_j_t = theta[:, j].transpose()
        theta_j_t_x_i = theta_j_t[m] + theta_j_t[bias]
        denominator = denominator + math.exp(theta_j_t_x_i)

    t = -(indicator - (numerator/ denominator))
    j_theta[m], j_theta[bias] = t, t
    #print i, k, indicator, numerator, denominator, numerator / denominator, m, j_theta
    return j_theta


def calculate_likelihood(theta, feature_vector, labels, labels_unique, words_unique):

    likelihood = 0.0
    for i in xrange(0, len(labels)):
        for k in xrange(0, len(labels_unique)):
            if labels[i] == labels_unique[k]:
                indicator = 1
            else:
                indicator = 0

            m, bias = feature_vector[i], len(words_unique)
            theta_k_t = theta[:, k].transpose()
            theta_k_t_x_i = theta_k_t[m] + theta_k_t[bias]
            numerator = math.exp(theta_k_t_x_i)
            denominator = 0
            for j in xrange(0, len(labels_unique)):
                theta_j_t = theta[:, j].transpose()
                theta_j_t_x_i = theta_j_t[m] + theta_j_t[bias]
                denominator = denominator + math.exp(theta_j_t_x_i)
            likelihood = likelihood + (indicator * math.log(numerator/denominator))
    print round(-likelihood/len(labels), 6)
    return 0


train_tsv = sys.argv[1]
validation_tsv = sys.argv[2]
num_epoch = int(sys.argv[7])

train_words, train_labels, train_words_unique, train_labels_unique = parse_train_tsv(train_tsv)
train_feature_vector = construct_feature_vector(train_words, train_words_unique)
theta0 = calculate_probability(num_epoch,train_feature_vector, train_labels, train_labels_unique, train_words_unique)
likelihood0 = calculate_likelihood(theta0, train_feature_vector, train_labels, train_labels_unique, train_words_unique)

validation_words, validation_labels = parse_tsv(validation_tsv)
validation_feature_vector = construct_feature_vector(validation_words, train_words_unique)
likelihood0 = calculate_likelihood(theta0, validation_feature_vector, validation_labels, train_labels_unique, train_words_unique)


