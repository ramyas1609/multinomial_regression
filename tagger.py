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
            words.append("BlankLine")
            labels.append("BlankLine")
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
            words.append("BlankLine")
            labels.append("BlankLine")
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
        if word == "BlankLine":
            m = -1
        else:
            m = words_unique.index(word)
        feature_vector.append(m)
    return feature_vector


def calculate_probability(epoch, feature_vector, labels, labels_unique, words_unique):
    theta = np.zeros(shape=(len(words_unique) + 1, len(labels_unique)))
    for z in xrange(0, epoch):
        for i in xrange(0, len(labels)):
            if labels[i] == "BlankLine":
                continue
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

    t = -(indicator - (numerator / denominator))
    j_theta[m], j_theta[bias] = t, t
    #print i, k, indicator, numerator, denominator, numerator / denominator, m, j_theta
    return j_theta


def calculate_likelihood(theta, feature_vector, labels, labels_unique, words, words_unique):
    pd = {}
    likelihood = 0.0
    n = 0
    for i in xrange(0, len(labels)):
        if words[i] == "BlankLine":
            continue
        pd_i = []
        n = n + 1
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
            pd_i.append(numerator/denominator)
        pd[words[i]] = pd_i
    return round(-likelihood/n, 6), pd


def prediction(pd, words, labels, labels_unique):
    num_error = 0.0
    n = 0.0
    predicted_labels = []
    for word in words:
        if word == "BlankLine":
            predicted_labels.append("\n")
            continue
        pd_i = pd[word]
        max_p, index = 0, 0
        for i in xrange(0, len(pd_i)):
            if pd_i[i] > max_p:
                max_p = pd_i[i]
                index = i
        predicted_labels.append(labels_unique[index])
        n = n + 1
        if labels[words.index(word)] != labels_unique[index]:
            num_error = num_error + 1

    error = num_error/n
    return predicted_labels, "{:.6f}".format(round(error, 6))


def write_labels_out(labels, filename):
    labels_file = open(filename, "w")
    for label in labels:
        labels_file.write(label)
        if label != "\n":
            labels_file.write("\n")
    labels_file.write("\n")


def write_metrics_out(filename, train_likelihood_1, validation_likelihood_1,
                      train_likelihood_2, validation_likelihood_2,
                      train_error, test_error):

    metrics_file = open(filename, "w")
    metrics_file.write("Epoch=1 likelihood train: "+str(train_likelihood_1)+"\n")
    metrics_file.write("Epoch=1 likelihood validation: "+str(validation_likelihood_1)+"\n")
    metrics_file.write("Epoch=2 likelihood train: "+str(train_likelihood_2)+"\n")
    metrics_file.write("Epoch=2 likelihood validation: "+str(validation_likelihood_2)+"\n")
    metrics_file.write("Train error: "+str(train_error)+"\n")
    metrics_file.write("Test error: "+str(test_error))


train_tsv = sys.argv[1]
validation_tsv = sys.argv[2]
test_tsv = sys.argv[3]
train_out = sys.argv[4]
test_out = sys.argv[5]
metrics = sys.argv[6]
num_epoch = int(sys.argv[7])


train_words, train_labels, train_words_unique, train_labels_unique = parse_train_tsv(train_tsv)

train_feature_vector = construct_feature_vector(train_words, train_words_unique)

theta0 = calculate_probability(num_epoch, train_feature_vector, train_labels, train_labels_unique, train_words_unique)

train_likelihood, train_pd = calculate_likelihood(theta0, train_feature_vector, train_labels,
                                                  train_labels_unique, train_words,train_words_unique)

train_predicted_labels, train_error = prediction(train_pd, train_words, train_labels, train_labels_unique)

write_labels_out(train_predicted_labels, train_out)

validation_words, validation_labels = parse_tsv(validation_tsv)

validation_feature_vector = construct_feature_vector(validation_words, train_words_unique)

validation_likelihood, validation_pd = calculate_likelihood(theta0, validation_feature_vector, validation_labels,
                                                            train_labels_unique, train_words, train_words_unique)

test_words, test_labels = parse_tsv(test_tsv)

test_feature_vector = construct_feature_vector(test_words, train_words_unique)

test_likelihood, test_pd = calculate_likelihood(theta0, test_feature_vector, test_labels,
                                                train_labels_unique, test_words, train_words_unique)

test_predicted_labels, test_error = prediction(test_pd, test_words, test_labels, train_labels_unique)

write_labels_out(test_predicted_labels, test_out)

train_likelihood_1, validation_likelihood_1, train_likelihood_2, validation_likelihood_2 = 0.0, 0.0, 0.0, 0.0

if num_epoch == 1:
    train_likelihood_1 = train_likelihood
    validation_likelihood_1 = validation_likelihood
    theta2 = calculate_probability(int(2), train_feature_vector, train_labels, train_labels_unique,
                                   train_words_unique)

    train_likelihood_2, train_pd_2 = calculate_likelihood(theta2, train_feature_vector, train_labels,
                                                          train_labels_unique, train_words, train_words_unique)
    validation_likelihood_2, validation_pd_2 = calculate_likelihood(theta2, validation_feature_vector, validation_labels,
                                                                    train_labels_unique, train_words, train_words_unique)
elif num_epoch == 2:
    train_likelihood_2 = train_likelihood
    validation_likelihood_2 = validation_likelihood
    theta1 = calculate_probability(int(1), train_feature_vector, train_labels, train_labels_unique,
                                   train_words_unique)

    train_likelihood_1, train_pd_1 = calculate_likelihood(theta1, train_feature_vector, train_labels,
                                                          train_labels_unique, train_words, train_words_unique)
    validation_likelihood_1, validation_pd_1 = calculate_likelihood(theta1, validation_feature_vector, validation_labels,
                                                                    train_labels_unique, train_words, train_words_unique)

write_metrics_out(metrics, train_likelihood_1, validation_likelihood_1,
                  train_likelihood_2, validation_likelihood_2,
                  train_error, test_error)


