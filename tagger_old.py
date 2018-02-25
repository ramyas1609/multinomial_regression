import sys, math
import numpy as np


def parse_train_tsv(filename,model):
    words = []
    labels = []
    if model == int(2):
        words_unique = ["BOS", "EOS"]
    else:
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
    tsv_file.close()
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
    tsv_file.close()
    return words, labels


def construct_feature_vector(words, words_unique, model):
    feature_vector = []
    for i in xrange(0, len(words)):
        if model == int(2):
            if words[i] == "BlankLine":
                feature_vector.append(-1)
                continue
            if i == 0:
                m = [words_unique.index("BOS"), words_unique.index(words[i]), words_unique.index(words[i + 1])]
            elif i == len(words) - 1:
                m = [words_unique.index(words[i - 1]), words_unique.index(words[i]), words_unique.index("EOS")]
            else:
                if words[i-1] == "BlankLine":
                    m = [words_unique.index("BOS"),words_unique.index(words[i]), words_unique.index(words[i+1])]
                elif words[i+1] == "BlankLine":
                    m = [words_unique.index(words[i-1]), words_unique.index(words[i]), words_unique.index("EOS")]
                else:
                    m = [words_unique.index(words[i-1]), words_unique.index(words[i]), words_unique.index(words[i+1])]
        else:
            if words[i] == "BlankLine":
                m = -1
            else:
                m = words_unique.index(words[i])
        feature_vector.append(m)

    return feature_vector


def calculate_theta(epoch, feature_vector, labels, labels_unique, words_unique, model):
    if model == int(2):
        theta = np.zeros(shape=((3 * len(words_unique)) + 1, len(labels_unique)))
        temp_theta = np.zeros(shape=((3 * len(words_unique)) + 1, len(labels_unique)))
    else:
        theta = np.zeros(shape=(len(words_unique) + 1, len(labels_unique)))
        temp_theta = np.zeros(shape=(len(words_unique) + 1, len(labels_unique)))

    for z in xrange(0, epoch):
        for i in xrange(0, len(labels)):
            if labels[i] == "BlankLine":
                continue
            temp_theta.fill(0)
            for k in xrange(0, len(labels_unique)):
                j_theta = calculate_gradient(i, k, theta, feature_vector, labels, labels_unique, words_unique, model)
                temp_theta[:, k] = 0.5 * j_theta
            theta = theta - temp_theta
    return theta


def calculate_gradient(i, k, theta, feature_vector, labels, labels_unique, words_unique, model):
    if model == int(2):
        j_theta = np.zeros(shape=(3 * (len(words_unique)) + 1))
        bias = 3 * (len(words_unique))
    else:
        j_theta = np.zeros(shape=(len(words_unique) + 1))
        bias = len(words_unique)

    if labels[i] == labels_unique[k]:
        indicator = 1
    else:
        indicator = 0

    m = feature_vector[i]
    theta_k_t = theta[:, k]. transpose()
    if model == int(2):
        theta_k_t_x_i = theta_k_t[m[0]] + theta_k_t[m[1] + len(words_unique)] + theta_k_t[m[2] + (2 * len(words_unique))] + theta_k_t[bias]
    else:
        theta_k_t_x_i = theta_k_t[m] + theta_k_t[bias]
    numerator = math.exp(theta_k_t_x_i)
    denominator = 0
    for j in xrange(0, len(labels_unique)):
        theta_j_t = theta[:, j].transpose()
        if model == int(2):
            theta_j_t_x_i = theta_j_t[m[0]] + theta_j_t[m[1]+len(words_unique)] + theta_j_t[m[2]+(2*len(words_unique))] + theta_j_t[bias]
        else:
            theta_j_t_x_i = theta_j_t[m] + theta_j_t[bias]
        denominator = denominator + math.exp(theta_j_t_x_i)

    t = -(indicator - (numerator / denominator))
    if model == int(2):
        j_theta[m[0]], j_theta[m[1]+len(words_unique)], j_theta[m[2]+(2*len(words_unique))], j_theta[bias] = t, t, t, t
    else:
        j_theta[m], j_theta[bias] = t, t
    #print i, k, indicator, numerator, denominator, numerator / denominator, m, j_theta
    return j_theta


def calculate_likelihood(theta, feature_vector, labels, labels_unique, words, words_unique, model):
    likelihood = 0.0
    n = 0
    for i in xrange(0, len(words)):
        if words[i] == "BlankLine":
            continue
        n = n + 1
        m = feature_vector[i]
        if model == int(2):
            m[1] = m[1] + len(words_unique)
            m[2] = m[2] + (2 * len(words_unique))
            bias = 3*len(words_unique)
        else:
            bias = len(words_unique)

        for k in xrange(0, len(labels_unique)):
            if labels[i] == labels_unique[k]:
                indicator = 1
            else:
                indicator = 0

            theta_k_t = theta[:, k].transpose()
            if model == int(2):
                theta_k_t_x_i = theta_k_t[m[0]] + theta_k_t[m[1]] + theta_k_t[m[2]] + theta_k_t[bias]
            else:
                theta_k_t_x_i = theta_k_t[m] + theta_k_t[bias]
            numerator = math.exp(theta_k_t_x_i)
            denominator = 0
            for j in xrange(0, len(labels_unique)):
                theta_j_t = theta[:, j].transpose()
                if model == int(2):
                    theta_j_t_x_i = theta_j_t[m[0]] + theta_j_t[m[1]] + theta_j_t[m[2]] + theta_j_t[bias]
                else:
                    theta_j_t_x_i = theta_j_t[m] + theta_j_t[bias]
                denominator = denominator + math.exp(theta_j_t_x_i)
            likelihood = likelihood + (indicator * math.log(numerator/denominator))
    return "{:.6f}".format(round(-likelihood/n, 6))


def prediction_model_1(theta, words, words_unique, labels, labels_unique):
    pd = {}
    for i in xrange(0, len(words_unique)):
        pd_i = []
        m = i
        bias = len(words_unique)
        for k in xrange(0, len(labels_unique)):
            theta_k_t = theta[:, k].transpose()
            theta_k_t_x_i = theta_k_t[m] + theta_k_t[bias]
            numerator = math.exp(theta_k_t_x_i)
            denominator = 0
            for j in xrange(0, len(labels_unique)):
                theta_j_t = theta[:, j].transpose()
                theta_j_t_x_i = theta_j_t[m] + theta_j_t[bias]
                denominator = denominator + math.exp(theta_j_t_x_i)
            pd_i.append(numerator/denominator)
        pd[words_unique[i]] = pd_i

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

    error = num_error / n
    return predicted_labels, "{:.6f}".format(round(error, 6))


def prediction_model_2(theta, words, words_unique, labels, labels_unique, feature_vector):
    num_error = 0.0
    n = 0.0
    predicted_labels = []
    for i in xrange(0, len(words)):
        if words[i] == "BlankLine":
            predicted_labels.append("\n")
            continue
        bias = 3*len(words_unique)
        m = feature_vector[i]
        m[1] = m[0] + len(words_unique)
        m[2] = m[0] + 2*(len(words_unique))
        pd_i = []
        for k in xrange(0, len(labels_unique)):
            theta_k_t = theta[:, k].transpose()
            theta_k_t_x_i = theta_k_t[m[0]] + theta_k_t[m[1]] + theta_k_t[m[2]] + theta_k_t[bias]
            numerator = math.exp(theta_k_t_x_i)
            denominator = 0
            for j in xrange(0, len(labels_unique)):
                theta_j_t = theta[:, j].transpose()
                theta_j_t_x_i = theta_j_t[m[0]] + theta_j_t[m[1]] + theta_j_t[m[2]] + theta_j_t[bias]
                denominator = denominator + math.exp(theta_j_t_x_i)
            pd_i.append(numerator / denominator)

        max_p, index = 0, 0
        for i in xrange(0, len(pd_i)):
            if pd_i[i] > max_p:
                max_p = pd_i[i]
                index = i
        predicted_labels.append(labels_unique[index])
        n = n + 1
        if labels[words.index(words[i])] != labels_unique[index]:
            num_error = num_error + 1

    error = num_error / n
    return predicted_labels, "{:.6f}".format(round(error, 6))


def write_labels_out(labels, filename):
    labels_file = open(filename, "w")
    for label in labels:
        labels_file.write(label)
        if label != "\n":
            labels_file.write("\n")
    labels_file.write("\n")


def write_metrics_out(filename, train_likelihoods, validation_likelihoods, train_error, test_error):

    metrics_file = open(filename, "w")
    for i in xrange(0, len(t_l)):
        metrics_file.write("Epoch=" + str(i + 1) + " likelihood train: "+str(train_likelihoods[i])+"\n")
        metrics_file.write("Epoch=" + str(i + 1) + " likelihood validation: " + str(validation_likelihoods[i]) + "\n")

    metrics_file.write("Train error: "+str(train_error)+"\n")
    metrics_file.write("Test error: "+str(test_error)+"\n")
    metrics_file.close()


train_tsv = sys.argv[1]
validation_tsv = sys.argv[2]
test_tsv = sys.argv[3]
train_out = sys.argv[4]
test_out = sys.argv[5]
metrics = sys.argv[6]
num_epoch = int(sys.argv[7])
model_number = int(sys.argv[8])

train_words, train_labels, train_words_unique, train_labels_unique = parse_train_tsv(train_tsv, model_number)
train_feature_vector = construct_feature_vector(train_words, train_words_unique, model_number)

validation_words, validation_labels = parse_tsv(validation_tsv)
validation_feature_vector = construct_feature_vector(validation_words, train_words_unique, model_number)

t_l = []
v_l = []

for ne in xrange(1, num_epoch+1):

    theta0 = calculate_theta(ne, train_feature_vector, train_labels, train_labels_unique, train_words_unique,
                             model_number)
    train_likelihood = calculate_likelihood(theta0, train_feature_vector, train_labels,
                                            train_labels_unique, train_words, train_words_unique, model_number)

    t_l.append(train_likelihood)
    if model_number == int(1):
        train_predicted_labels, train_error = prediction_model_1(theta0, train_words, train_words_unique, train_labels,
                                                                 train_labels_unique)
    else:
        train_predicted_labels, train_error = prediction_model_2(theta0, train_words, train_words_unique, train_labels,
                                                                 train_labels_unique, train_feature_vector)
        print train_predicted_labels, train_error

    validation_likelihood = calculate_likelihood(theta0, validation_feature_vector, validation_labels,
                                                 train_labels_unique, validation_words, train_words_unique, model_number)

    v_l.append(validation_likelihood)


write_labels_out(train_predicted_labels, train_out)

test_words, test_labels = parse_tsv(test_tsv)
test_feature_vector = construct_feature_vector(test_words, train_words_unique, model_number)

test_likelihood = calculate_likelihood(theta0, test_feature_vector, test_labels,
                                       train_labels_unique, test_words, train_words_unique, model_number)
if model_number == int(1):
    test_predicted_labels, test_error = prediction_model_1(theta0, test_words, train_words_unique, test_labels, train_labels_unique)
else:
    test_predicted_labels, test_error = prediction_model_2(theta0, test_words, train_words_unique, test_labels,
                                                           train_labels_unique, train_feature_vector)

write_labels_out(test_predicted_labels, test_out)


write_metrics_out(metrics, t_l, v_l, train_error, test_error)
