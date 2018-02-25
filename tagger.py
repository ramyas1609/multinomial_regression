from math import *
import numpy as np
import sys

train_words = []
train_labels = []
train_words_unique = []
train_labels_unique = []
train_feature_vector = []
t_l = []
train_error = None

validation_words =[]
validation_labels = []
validation_feature_vector = []
v_l = []

test_words = []
test_labels = []
test_feature_vector = []
test_error = None


def parse_train_tsv(filename):
    tsv_file = open(filename, "r")

    for line in tsv_file.readlines():
        if line == "\n":
            train_words.append("BlankLine")
            train_labels.append("BlankLine")
            continue

        elements = line.rstrip().split("\t")
        word = elements[0]
        label = elements[1]

        train_words.append(word)
        if word not in train_words_unique:
            train_words_unique.append(word)

        train_labels.append(label)
        if label not in train_labels_unique:
            train_labels_unique.append(label)

    train_labels_unique.sort()


def parse_tsv(filename, type):
    if type == "validation":
        words = validation_words
        labels = validation_labels
    elif type == "test":
        words = test_words
        labels = test_labels

    tsv_file = open(filename, "r")
    for line in tsv_file.readlines():
        if line == "\n":
            words.append("BlankLine")
            labels.append("BlankLine")
            continue

        elements = line.rstrip().split("\t")
        words.append(elements[0])
        labels.append(elements[1])
    tsv_file.close()


def construct_feature_vector(model, type):
    if type == "train":
        train_words_unique.append("BOS")
        train_words_unique.append("EOS")
        words = train_words
        feature_vector = train_feature_vector
    elif type == "validation":
        words = validation_words
        feature_vector = validation_feature_vector
    elif type == "test":
        words = test_words
        feature_vector = test_feature_vector

    if model == 2:
        tw_l = len(words)
        for i in xrange(0, tw_l):
            if words[i] == "BlankLine":
                feature_vector.append(-1)
                continue
            if i == 0:
                m = [train_words_unique.index("BOS"), train_words_unique.index(words[i]), train_words_unique.index(words[i + 1])]
            elif i == tw_l - 1:
                m = [train_words_unique.index(words[i - 1]), train_words_unique.index(words[i]), train_words_unique.index("EOS")]
            else:
                if words[i - 1] == "BlankLine":
                    m = [train_words_unique.index("BOS"), train_words_unique.index(words[i]), train_words_unique.index(words[i + 1])]
                elif words[i + 1] == "BlankLine":
                    m = [train_words_unique.index(words[i - 1]), train_words_unique.index(words[i]), train_words_unique.index("EOS")]
                else:
                    m = [train_words_unique.index(words[i - 1]), train_words_unique.index(words[i]), train_words_unique.index(words[i + 1])]
            feature_vector.append(m)
    else:
        for word in words:
            if word == "BlankLine":
                m = -1
            else:
                m = train_words_unique.index(word)
            feature_vector.append(m)


def calculate_theta_1():

    theta = np.zeros(shape=(len(train_labels_unique), len(train_words_unique) + 1))
    temp_theta = np.zeros(shape=(len(train_labels_unique), len(train_words_unique) + 1))

    twu_l = len(train_words_unique)
    tl_l = len(train_labels)
    tlu_l = len(train_labels_unique)

    bias = twu_l

    for z in xrange(0, num_epoch):

        for i in xrange(0, tl_l):

            if train_labels[i] == "BlankLine":
                continue

            temp_theta.fill(0)
            m = train_feature_vector[i]

            denominator = 0
            for j in xrange(0, tlu_l):
                denominator = denominator + exp(theta[j][m] + theta[j][bias])

            for k in xrange(0, tlu_l):
                if train_labels[i] == train_labels_unique[k]:
                    indicator = 1
                else:
                    indicator = 0

                numerator = exp(theta[k][m] + theta[k][bias])
                t = -(indicator - (numerator / denominator))
                temp_theta[k][m], temp_theta[k][bias] = 0.5 * t, 0.5 * t
            theta = theta - temp_theta
        t_l.append(calculate_train_likelihood_1(theta))
        v_l.append(calculate_validation_likelihood_1(theta))
    return theta


def calculate_train_likelihood_1(theta):
    likelihood = 0.0
    n = 0
    i0 = len(train_labels)
    k0 = len(train_labels_unique)
    w0 = len(train_words_unique)
    bias = w0
    for i in xrange(0, i0):
        if train_words[i] == "BlankLine":
            continue
        n = n + 1
        m = train_feature_vector[i]

        denominator = 0
        for j in xrange(0, k0):
            denominator = denominator + exp(theta[j][m] + theta[j][bias])

        for k in xrange(0, k0):
            if train_labels[i] != train_labels_unique[k]:
                continue
            numerator = exp(theta[k][m] + theta[k][bias])
            likelihood = likelihood + (log(numerator / denominator))

    return "{:.6f}".format(round(-likelihood / n, 6))


def calculate_validation_likelihood_1(theta):
    likelihood = 0.0
    n = 0
    i0 = len(validation_labels)
    k0 = len(train_labels_unique)
    w0 = len(train_words_unique)
    bias = w0
    for i in xrange(0, i0):
        if validation_words[i] == "BlankLine":
            continue
        n = n + 1
        m = validation_feature_vector[i]

        denominator = 0
        for j in xrange(0, k0):
            denominator = denominator + exp(theta[j][m] + theta[j][bias])

        for k in xrange(0, k0):
            if validation_labels[i] != train_labels_unique[k]:
                continue
            numerator = exp(theta[k][m] + theta[k][bias])
            likelihood = likelihood + (log(numerator / denominator))

    return "{:.6f}".format(round(-likelihood / n, 6))


def calculate_theta_2():

    theta = np.zeros(shape=(len(train_labels_unique), 3 * len(train_words_unique) + 1))
    temp_theta = np.zeros(shape=(len(train_labels_unique), 3 * len(train_words_unique) + 1))

    twu_l = len(train_words_unique)
    tl_l = len(train_labels)
    tlu_l = len(train_labels_unique)

    bias = 3 * twu_l

    for z in xrange(0, num_epoch):

        for i in xrange(0, tl_l):

            if train_labels[i] == "BlankLine":
                continue

            temp_theta.fill(0)
            m = train_feature_vector[i]
            m0 = m[0]
            m1 = m[1] + twu_l
            m2 = m[2] + (2 * twu_l)

            denominator = 0
            for j in xrange(0, tlu_l):
                denominator = denominator + exp(theta[j][m0] + theta[j][m1] + theta[j][m2] + theta[j][bias])

            for k in xrange(0, tlu_l):
                if train_labels[i] == train_labels_unique[k]:
                    indicator = 1
                else:
                    indicator = 0

                numerator = exp(theta[k][m0] + theta[k][m1] + theta[k][m2] + theta[k][bias])
                t = -(indicator - (numerator / denominator))
                temp_theta[k][m0], temp_theta[k][m1], temp_theta[k][m2], temp_theta[k][bias] = 0.5 * t, 0.5 * t, 0.5 * t, 0.5 * t

            theta = theta - temp_theta

        t_l.append(calculate_train_likelihood_2(theta))
        v_l.append(calculate_validation_likelihood_2(theta))
    return theta


def calculate_train_likelihood_2(theta):
    likelihood = 0.0
    n = 0
    tl_l = len(train_labels)
    tlu_l = len(train_labels_unique)
    twu_l = len(train_words_unique)
    bias = 3 * twu_l

    for i in xrange(0, tl_l):
        if train_words[i] == "BlankLine":
            continue
        n = n + 1
        m = train_feature_vector[i]
        m0 = m[0]
        m1 = m[1] + twu_l
        m2 = m[2] + (2 * twu_l)

        denominator = 0
        for j in xrange(0, tlu_l):
            denominator = denominator + exp(theta[j][m0] + theta[j][m1] + theta[j][m2] + theta[j][bias])

        for k in xrange(0, tlu_l):
            if train_labels[i] != train_labels_unique[k]:
                continue
            numerator = exp(theta[k][m0] + theta[k][m1] + theta[k][m2] + theta[k][bias])
            likelihood = likelihood + (log(numerator / denominator))

    return "{:.6f}".format(round(-likelihood / n, 6))


def calculate_validation_likelihood_2(theta):
    likelihood = 0.0
    n = 0
    vl_l = len(validation_labels)
    tlu_l = len(train_labels_unique)
    twu_l = len(train_words_unique)
    bias = 3 * twu_l

    for i in xrange(0, vl_l):
        if validation_words[i] == "BlankLine":
            continue
        n = n + 1
        m = validation_feature_vector[i]
        m0 = m[0]
        m1 = m[1] + twu_l
        m2 = m[2] + (2 * twu_l)

        denominator = 0
        for j in xrange(0, tlu_l):
            denominator = denominator + exp(theta[j][m0] + theta[j][m1] + theta[j][m2] + theta[j][bias])

        for k in xrange(0, tlu_l):
            if validation_labels[i] != train_labels_unique[k]:
                continue
            numerator = exp(theta[k][m0] + theta[k][m1] + theta[k][m2] + theta[k][bias])
            likelihood = likelihood + (log(numerator / denominator))

    return "{:.6f}".format(round(-likelihood / n, 6))


def predict_train_1(theta, filename):
    num_error = 0
    num_samples = 0
    t_l_f = open(filename, "w")
    tw_l = len(train_words)
    tlu_l = len(train_labels_unique)
    bias = len(train_words_unique)
    for i in xrange(0, tw_l):
        if train_words[i] == "BlankLine":
            t_l_f.write("\n")
            continue
        num_samples = num_samples + 1
        max_p = 0.0
        max_index = 0
        m = train_feature_vector[i]
        denominator = 0
        for j in xrange(0, tlu_l):
            denominator = denominator + exp(theta[j][m] + theta[j][bias])
        for k in xrange(0, tlu_l):
            numerator = exp(theta[k][m] + theta[k][bias])
            p = numerator/denominator
            if p > max_p:
                max_p = p
                max_index = k
        t_l_f.write(train_labels_unique[max_index]+"\n")
        if train_labels[i] != train_labels_unique[max_index]:
            num_error = num_error + 1
    t_l_f.write("\n")
    t_l_f.close()
    return "{:.6f}".format(round(num_error*1.0/num_samples, 6))


def predict_test_1(theta, filename):
    num_error = 0
    num_samples = 0
    t_l_f = open(filename, "w")
    tw_l = len(test_words)
    tlu_l = len(train_labels_unique)
    bias = len(train_words_unique)
    for i in xrange(0, tw_l):
        if test_words[i] == "BlankLine":
            t_l_f.write("\n")
            continue
        num_samples = num_samples + 1
        max_p = 0.0
        max_index = 0
        m = test_feature_vector[i]
        denominator = 0
        for j in xrange(0, tlu_l):
            denominator = denominator + exp(theta[j][m] + theta[j][bias])
        for k in xrange(0, tlu_l):
            numerator = exp(theta[k][m] + theta[k][bias])
            p = numerator/denominator
            if p > max_p:
                max_p = p
                max_index = k
        t_l_f.write(train_labels_unique[max_index]+"\n")
        if test_labels[i] != train_labels_unique[max_index]:
            num_error = num_error + 1
    t_l_f.write("\n")
    t_l_f.close()
    return "{:.6f}".format(round(num_error*1.0/num_samples, 6))


def predict_train_2(theta, filename):
    num_error = 0
    num_samples = 0
    t_l_f = open(filename, "w")
    twu_l = len(train_words_unique)
    tw_l = len(train_words)
    tlu_l = len(train_labels_unique)
    bias = 3 * twu_l
    for i in xrange(0, tw_l):
        if train_words[i] == "BlankLine":
            t_l_f.write("\n")
            continue
        num_samples = num_samples + 1
        max_p = 0.0
        max_index = 0
        m = train_feature_vector[i]
        m0 = m[0]
        m1 = m[1] + twu_l
        m2 = m[2] + (2 * twu_l)
        denominator = 0
        for j in xrange(0, tlu_l):
            denominator = denominator + exp(theta[j][m0] + theta[j][m1] + theta[j][m2] + theta[j][bias])
        for k in xrange(0, tlu_l):
            numerator = exp(theta[k][m0] + theta[k][m1] + theta[k][m2] + theta[k][bias])
            p = numerator/denominator
            if p > max_p:
                max_p = p
                max_index = k
        t_l_f.write(train_labels_unique[max_index]+"\n")
        if train_labels[i] != train_labels_unique[max_index]:
            num_error = num_error + 1
    t_l_f.write("\n")
    t_l_f.close()
    return "{:.6f}".format(round(num_error*1.0/num_samples, 6))


def predict_test_2(theta, filename):
    num_error = 0
    num_samples = 0
    t_l_f = open(filename, "w")
    twu_l = len(train_words_unique)
    tw_l = len(test_words)
    tlu_l = len(train_labels_unique)
    bias = 3 * twu_l
    for i in xrange(0, tw_l):
        if test_words[i] == "BlankLine":
            t_l_f.write("\n")
            continue
        num_samples = num_samples + 1
        max_p = 0.0
        max_index = 0
        m = test_feature_vector[i]
        m0 = m[0]
        m1 = m[1] + twu_l
        m2 = m[2] + (2 * twu_l)
        denominator = 0
        for j in xrange(0, tlu_l):
            denominator = denominator + exp(theta[j][m0] + theta[j][m1] + theta[j][m2] + theta[j][bias])
        for k in xrange(0, tlu_l):
            numerator = exp(theta[k][m0] + theta[k][m1] + theta[k][m2] + theta[k][bias])
            p = numerator/denominator
            if p > max_p:
                max_p = p
                max_index = k
        t_l_f.write(train_labels_unique[max_index]+"\n")
        if test_labels[i] != train_labels_unique[max_index]:
            num_error = num_error + 1
    t_l_f.write("\n")
    t_l_f.close()
    return "{:.6f}".format(round(num_error*1.0/num_samples, 6))


def write_metrics(filename):
    metrics_file = open(filename, "w")
    for i in xrange(0, len(t_l)):
        metrics_file.write("Epoch=" + str(i + 1) + " likelihood train: "+str(t_l[i])+"\n")
        metrics_file.write("Epoch=" + str(i + 1) + " likelihood validation: " + str(v_l[i]) + "\n")

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


parse_train_tsv(train_tsv)
construct_feature_vector(model_number, "train")

parse_tsv(validation_tsv, "validation")
construct_feature_vector(model_number, "validation")

parse_tsv(test_tsv,"test")
construct_feature_vector(model_number, "test")

if model_number == 1:
    t1 = calculate_theta_1()
    train_error = predict_train_1(t1, train_out)
    test_error = predict_test_1(t1, test_out)
else:
    t2 = calculate_theta_2()
    train_error = predict_train_2(t2, train_out)
    test_error = predict_test_2(t2, test_out)

write_metrics(metrics)


