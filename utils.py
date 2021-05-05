# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 17:41:19 2017
"""

from brian2 import *
import csv
import os
import numpy as np
import seaborn as sns
import pandas as pd
import pickle
from os import listdir
from os.path import isfile, join
import re
import time
import errno

DATA_PATH = './data'
BATCH_FOLDER = './data'

n_rate = 16.0

dimensions_dict = {"second": get_or_create_dimension(time=1),
                   "volt": get_or_create_dimension(length=2, mass=1, time=-3, current=-1)}


def image_to_spike(image, factor, dropout=None):
    freq = []
    dropout_matrix = None
    if dropout is not None:
        dropout_matrix = numpy.random.choice([0, 1], size=28 * 28, p=[dropout, 1 - dropout])
    for idx in range(0, len(image)):
        drop = 1
        if dropout is not None:
            drop = dropout_matrix[idx]
        firering_rate = image[idx] * factor * drop
        freq.append(int(firering_rate))
    return freq


def image_to_spike_noise(image, factor, noise):
    freq = []
    for idx in range(0, len(image)):
        firering_rate = image[idx] * factor + 255 * factor*random()*noise
        freq.append(int(firering_rate))
    return freq


def get_n_images(n: int, factor, egdes=False, dropout=None):
    image_train_path = os.path.join(DATA_PATH, 'train-images-idx3-ubyte')
    lables_train_path = os.path.join(DATA_PATH, 'train-labels-idx1-ubyte')
    results_raw = get_data_and_labels(image_train_path, lables_train_path, egdes)
    counter = [0] * 10
    results = []
    for row in results_raw:
        freq = image_to_spike(row[1], factor, dropout)
        label = row[0]
        if counter[row[0]] < n:
            results.append((label, freq))
            counter[label] += 1
        if len(results) == n * 10:
            break
    return results


def get_batch_in_range(start, end, factor, prefix=''):
    image_train_path = os.path.join(DATA_PATH, 'train-images-idx3-ubyte')
    lables_train_path = os.path.join(DATA_PATH, 'train-labels-idx1-ubyte')
    results_raw = get_data_and_labels(image_train_path, lables_train_path)[start:end]

    results = []

    for row in results_raw:
        freq = image_to_spike(row[1], factor)
        label = row[0]
        results.append((label, freq))

    batch = prepare_timing_batch(0, 400, 200, results, dt=0.5)
    save_batch(batch, "batch_range" + prefix, BATCH_FOLDER)

    return batch


def get_images_by_id(ids: array, factor, test_batch=False):
    if test_batch:
        image_path = os.path.join(DATA_PATH, 't10k-images.idx3-ubyte')
        lables_path = os.path.join(DATA_PATH, 't10k-labels.idx1-ubyte')
    else:
        image_path = os.path.join(DATA_PATH, 'train-images.idx3-ubyte')
        lables_path = os.path.join(DATA_PATH, 'train-labels.idx1-ubyte')

    results_raw = get_data_and_labels(image_path, lables_path)

    results = []
    for id in ids:
        image = results_raw[id]
        results.append((image[0], image_to_spike(image[1], factor)))

    return results


def get_raw_images_by_ids(ids: array, test_batch=False):
    if test_batch:
        image_path = os.path.join(DATA_PATH, 't10k-images.idx3-ubyte')
        lables_path = os.path.join(DATA_PATH, 't10k-labels.idx1-ubyte')
    else:
        image_path = os.path.join(DATA_PATH, 'train-images.idx3-ubyte')
        lables_path = os.path.join(DATA_PATH, 'train-labels.idx1-ubyte')

    all_images = get_data_and_labels(image_path, lables_path)
    image_raw = []

    for id in ids:
        image_raw.append(all_images[id])

    return image_raw


def generate_selected_batch(ids, time=400, factor=0.1):
    result = get_images_by_id(ids, factor)
    batch = prepare_timing_batch(0, time, 200, result, dt=0.5)
    save_batch(batch, "batch_selected", BATCH_FOLDER)
    return batch

def generate_random_batch(ids, time=400, factor=0.1):
    result = get_images_by_id(ids, factor)
    batch = prepare_timing_batch(0, time, 200, result, dt=0.5)
    save_batch(batch, "batch_selected", BATCH_FOLDER)
    return batch


def svae_spike_images(file_name, indicies, times):
    csv_writer = csv.writer(open(file_name, 'w'))
    csv_writer.writerow(indices)
    csv.writer.writerow(times)


def get_first_ten_from_file(file_name):
    csv_reader = csv.reader(open(file_name, 'r'))


def prepare_timing_batch(start_time, exposition_time, rest_time, images, dt=0.1):
    indicies = []
    times = []
    batch = {'indicies': [],
             'times': [],
             'N': len(images),
             'sim_time': 0,
             'labels': [],
             'freq': [],
             'start_time': 0,
             'exposition_time': exposition_time,
             's_indicies': [],
             's_timings': [],
             'rest_time': rest_time}
    # start after rest
    current_time = start_time;
    s_indicies = []
    s_timings = []
    iii = 0
    for image in images:
        if iii % 1000 == 0:
            print(str(100*iii / len(images)) + '%')
        current_time += rest_time
        freqencies = image[1]
        s_indicies.append(image[0])
        s_timings.append(current_time)
        batch.get('labels').append(image[0])
        batch.get('freq').append(image[1])
        for i, input_freq in enumerate(freqencies, 0):
            if input_freq == 0:
                continue
            prop = (input_freq / 1000) * dt
            for t in range(0, int(exposition_time / dt)):
                if random() <= prop:
                    indicies.append(i)
                    times.append(round(dt * t + current_time, 1))
        current_time += exposition_time
        iii += 1
    batch['indicies'] = indicies
    batch['times'] = times * ms
    batch['sim_time'] = (exposition_time + rest_time) * batch['N'] * ms
    batch['s_indicies'] = s_indicies
    batch['s_times'] = s_timings * ms
    return batch


def save_batch(batch, name, folder=''):
    FILE = folder + '/' + name
    if not os.path.exists(os.path.dirname(FILE)):
        try:
            os.makedirs(os.path.dirname(FILE))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    with open(FILE, 'wb') as fp:
        pickle.dump(batch, fp)


def load_batch(name: object, folder=None) -> object:
    batch = {}
    file_name = folder + '/' + name if folder is not None else name
    with open(file_name, 'rb') as fp:
        batch = pickle.load(fp)

    return batch


def read_mnist(test=False):
    train_file = os.path.join(DATA_PATH, 'train.csv')
    if (test):
        train_file = os.path.join(DATA_PATH, 'test.csv')
    mnist_file = open(train_file, "r")
    reader = csv.reader(mnist_file)
    next(reader, None)
    return reader


def process_mnist():
    reader = read_mnist()
    results = []
    for row in reader:
        row_int = [int(elem) for elem in row]
        freq = image_to_spike(row_int[1:785], 0.1)
        label = row_int[0]
        results.append((label, freq))
    return results


def calculate_fireing_rate(img):
    sum = 0
    for rate in img:
        sum += rate
    sum = sum / (28 * 28)
    return sum


def generate_random_weights(stim_N: int, ext_N: int, wmax=1):
    return np.random.rand(stim_N, ext_N) * wmax

def generate_random_weights_asymetric(M, wmax=1):
    weights = np.random.rand(M) * wmax
    # for k in range(0, M - 1):
    #     weights.append(np.random.rand(M-1-k)*wmax)
    return weights


def calculate_match_vector(size: int, weigths: list) -> list:
    spike_nbr = len(weigths)
    match_vector = [0] * size
    if spike_nbr == 0:
        return match_vector
    for value in weigths:
        match_vector[value] += 1
    match_vector = [k / spike_nbr for k in match_vector]
    return array(match_vector)


def write_results(results: dict):
    file = open('wyniki.csv', 'w')
    csv_writer = csv.writer(file)
    for key in results:
        line = []
        line.append(key)
        line.extend(results.get(key))
        csv_writer.writerow(line)
    file.close()


def spike_result_array(exposition_time, rest_time, N, spike_trains, input_vector_nbr: int):
    res = {}
    results = [[0] * len(spike_trains) for i in range(0, N)]
    for neuron_nbr in spike_trains:
        neuron_spike_times = spike_trains.get(neuron_nbr)
        for spike_time in neuron_spike_times:
            image_nbr = int(floor(spike_time * 1000 / (exposition_time + rest_time)))
            results[image_nbr][neuron_nbr] += 1
    res['results'] = results
    res['spike_trains'] = spike_trains
    return res


def vector_result(spike_trains, time_from: int):
    results = [0] * len(spike_trains)
    for neuron_nbr in spike_trains:
        neuron_spike_times = spike_trains.get(neuron_nbr)
        for spike_time in neuron_spike_times:
            if spike_time < time_from * ms:
                continue
            results[neuron_nbr] += 1
    return results;


def read_pattern_vectors(file_name):
    results = {}
    file = open(file_name, 'r')
    csv_reader = csv.reader(file)
    for row in csv_reader:
        results[int(row[0])] = [int(k) for k in row[1:]]
    return results


def find_match(result, result_vec_dict, size):
    match = calculate_match_vector(size, result)
    best = -1
    winner = -1
    for key in result_vec_dict:
        temp_match = calculate_match_vector(size, result_vec_dict.get(key))
        product = dot(match, temp_match)
        if product > winner:
            winner = product
            best = key
    return best


def norm(vec):
    suma = 0
    for value in vec:
        suma += value * value
    return [float(x) / sqrt(suma) for x in vec]


def norm2(ref_dict, key):
    avg = [0] * len(ref_dict.get(key))
    for i in ref_dict:
        for k, val in enumerate(ref_dict.get(i), 0):
            avg[k] += val
    avg = [float(k) / len(ref_dict) for k in avg]

    vec = ref_dict.get(key)

    for i, val in enumerate(vec, 0):
        vec[i] = val - avg[i]
    return norm(vec)


def freq_base_ref(ref_dict):
    avg = [0] * len(ref_dict.get(0))
    freq = [0] * len(ref_dict.get(0))
    result_dict = {}
    for i in range(0, len(ref_dict.get(0))):
        for key in ref_dict:
            avg[i] += ref_dict.get(key)[i]
            if ref_dict.get(key)[i] > 0:
                freq[i] += 1
        if freq[i] > 0:
            avg[i] = avg[i] / freq[i]
    for key in ref_dict:
        result_vec = [0] * len(ref_dict.get(key))
        for i, val in enumerate(ref_dict.get(key), 0):
            result_vec[i] = 0 if val < avg[i] * 0.8 else 1
        result_dict[key] = result_vec
    return result_dict


def save_results(file_name, results):
    file = open(file_name, 'w')
    csv_writer = csv.writer(file)
    for result in results:
        csv_writer.writerow(result)


def load_results(file_name):
    res_file = open(file_name, 'r')
    results = []
    for line in res_file.readlines():
        results.append([float(el) for el in line.split(',')])
    return results


def predict_scalar_product(firing_vector, firing_for_label_dict):
    norm_vec = norm(firing_vector)
    best = -1
    winner_label = -1
    product_for_class = [0] * 10
    for label in firing_for_label_dict:
        norm_ref_vec = norm(firing_for_label_dict.get(label))
        scalar_product = dot(array(norm_vec), array(norm_ref_vec))
        product_for_class[label] = scalar_product
        if scalar_product > best:
            winner_label = label
            best = scalar_product
    return winner_label, product_for_class


def predict_neuron_class(firing_vector, neuron_class_maping):
    firing_for_class = [0] * 10
    lbl = -1
    for neuron_idx, spike_nbr in enumerate(firing_vector, 0):
        firing_for_class[neuron_class_maping.get(neuron_idx)] += spike_nbr
    if max(firing_for_class) > 0:
        lbl = firing_for_class.index(max(firing_for_class))
    return lbl, firing_for_class


def map_neuron_to_class(firing_for_label_dict):
    df = pd.DataFrame.from_dict(firing_for_label_dict, orient='index')
    neuron_class_mapping = {}
    for column in df:
        idx = df[column].argmax()
        neuron_class_mapping[column] = idx
    return neuron_class_mapping


def load_weights(file_name):
    dict_weights = {}
    file = open(file_name, 'r')
    lines = file.readlines()
    for i, line in enumerate(lines, 0):
        weights = lines[i].split(',')[0:784]
        weights = [float(i) for i in weights]
        dict_weights[i] = weights
    return dict_weights


def weights_dict(weights):
    weights_d = {}
    i = 0
    while i < int(weights.shape[0]):
        for j in range(0, int(weights.shape[0] / 784)):
            w = weights[i + j]
            weights_d[j] = [w] if j not in weights_d else weights_d.get(j) + [w]
        i = i + int(weights.shape[0] / 784)
    return weights_d


def print_weights(weights, maximum=100000000):
    print(weights.shape[0])
    weights_d = {}
    i = 0
    while i < int(weights.shape[0]):
        for j in range(0, int(weights.shape[0] / 784)):
            w = weights[i + j]
            weights_d[j] = [w] if j not in weights_d else weights_d.get(j) + [w]
        i = i + int(weights.shape[0] / 784)
    i = 0
    for key in weights_d:
        print(key)
        sns.heatmap(array(array(weights_d.get(key)).reshape(28, 28)))
        show()
        i += 1
        if (i > maximum):
            break

def get_n_weigths(weights,nbr):
    weights_d = {}
    i = 0
    while i < int(weights.shape[0]):
        for j in range(0, nbr):
            w = weights[i + j]
            weights_d[j] = [w] if j not in weights_d else weights_d.get(j) + [w]
        i = i + int(weights.shape[0] / 784)
    return weights_d



def print_weights2(weight_dict):
    for key in weight_dict:
        dic = [array([0]) if len(v) is 0 else v for v in weight_dict.get(key)]
        print(dic)
        sns.heatmap(array(dic).reshape(28, 28))
        show()


def generate_trianing_batch(n, repeat, factor=0.1, prefix='', edges=False, dropout=None):
    result = []
    print(os.listdir())
    first_ten = get_n_images(n, factor, edges, dropout) if dropout is not None else get_n_images(n, factor, edges)
    for i in range(0, repeat):
        result.extend(first_ten)
    batch = prepare_timing_batch(0, 400, 200, result, dt=0.5)
    print(BATCH_FOLDER)
    if dropout is not None:
        prefix = 'dropout_' + str(dropout)

    save_batch(batch, str(n) + "_" + str(repeat) + "batch_" + prefix, BATCH_FOLDER)
    return batch



def generate_test_batch(prefix='', factor=0.1, factor_list=None):
    results = []
    mnist = get_data_and_labels(os.path.join(DATA_PATH, 't10k-images-idx3-ubyte'),
                                os.path.join(DATA_PATH, 't10k-labels-idx1-ubyte'))
    for idx, row in enumerate(mnist, 0):
        if factor_list is not None:
            factor = factor_list[idx]
        freq = image_to_spike(row[1], factor)
        label = row[0]
        results.append((label, freq))
    batch = prepare_timing_batch(0, 400, 200, results, dt=0.5)
    save_batch(batch, prefix + "test_" + "batch", BATCH_FOLDER)
    return batch


def generate_test_batch_with_mapping(mapping, prefix='', factor=0.1):
    results = []
    mnist = get_data_and_labels(os.path.join(DATA_PATH, 't10k-images-idx3-ubyte'),
                                os.path.join(DATA_PATH, 't10k-labels-idx1-ubyte'))
    for idx, row in enumerate(mnist, 0):
        alpha = factor
        if idx in mapping.keys():
            alpha = factor + 0.05 * (4 - mapping.get(idx))
        freq = image_to_spike(row[1], alpha)
        label = row[0]
        results.append((label, freq))
    batch = prepare_timing_batch(0, 400, 200, results, dt=0.5)
    save_batch(batch, prefix + "test_" + "batch", BATCH_FOLDER)
    return batch


def generate_test_batch_idxs(prefix='', mapping={}, factor=0.1):
    results = []
    mnist = get_data_and_labels(os.path.join(DATA_PATH, 't10k-images.idx3-ubyte'),
                                os.path.join(DATA_PATH, 't10k-labels.idx1-ubyte'))
    for idx, row in enumerate(mnist, 0):
        if idx in mapping.keys():
            freq = image_to_spike(row[1], 0.1 + 0.05 * (4 - mapping.get(idx)));
            label = row[0]
            results.append((label, freq))
    batch = prepare_timing_batch(0, 400, 200, results, dt=0.5)
    save_batch(batch, prefix + "test_" + "batch", BATCH_FOLDER)
    return batch


def generate_frequency_matrix(first_ten: int, frequency_factor: float = 0.1):
    get_n_images(first_ten, frequency_factor)

def evaluate_neurons(labels, simulation_results, identify_batch):
    neuron_class_mapping = identify_batch.get('neuron_class_mapping')
    firing_for_label_dict = identify_batch.get('firing_for_label_dict')
    res_hist_vec = [0] * 10
    fal_hist_vec = [0] * 10
    res_hist_neural = [0] * 10
    fal_hist_neural = [0] * 10
    scalar_match = 0
    neuron_match = 0
    false_results = []
    total_results = []
    total_scalar_results = []
    label_freq = [0] * 10
    for k in labels:
        label_freq[k] += 1
    pred_label = []
    test_label = []
    pred_label_vec = []
    for i, firing_response_vec in enumerate(simulation_results.get('results'), 0):
        label = labels[i]
        scalar_pred = predict_scalar_product(firing_response_vec, firing_for_label_dict)
        neuron_pred = predict_neuron_class(firing_response_vec, neuron_class_mapping)
        total_results.append(neuron_pred[1])
        test_label.append(label)
        pred_label.append(neuron_pred[0])
        pred_label_vec.append(scalar_pred[0])
        total_scalar_results.append(scalar_pred[1])
        if scalar_pred[0] == label:
            res_hist_vec[label] += 1
            scalar_match += 1
        else:
            fal_hist_vec[scalar_pred[0]] += 1
        if label == neuron_pred[0]:
            res_hist_neural[label] += 1
            neuron_match += 1
        else:
            false_results.append(neuron_pred[1])

            fal_hist_neural[neuron_pred[0]] += 1
    neuron_for_class_hist = [0] * 10
    for neuron in neuron_class_mapping:
        neuron_for_class_hist[neuron_class_mapping.get(neuron)] += 1

    result_dict = {}
    result_dict['false_results'] = false_results
    result_dict['total_results'] = total_results
    result_dict['total_scalar_results'] = total_scalar_results
    result_dict['pred_vec'] = scalar_match / len(labels)
    result_dict['pred_neu'] = neuron_match / len(labels)
    result_dict['res_hist_vec'] = res_hist_vec
    result_dict['fal_hist_vec'] = fal_hist_vec
    result_dict['res_hist_neural'] = res_hist_neural
    result_dict['res_per_class_neural'] = [val / label_freq[i] for i, val in enumerate(res_hist_neural, 0)]
    result_dict['fal_hist_neural'] = fal_hist_neural
    result_dict['number_od_neuron_per_class'] = neuron_for_class_hist
    result_dict['pred_label'] = pred_label
    result_dict['scalar_label'] = pred_label_vec
    result_dict['test_label'] = test_label

    return result_dict


def get_data_and_labels(images_filename, labels_filename, edges=False, n=-1):
    print("Opening files ...",images_filename,labels_filename)
    images_file = open(images_filename, "rb")
    labels_file = open(labels_filename, "rb")

    try:
        print("Reading files ...")
        images_file.read(4)
        num_of_items = int.from_bytes(images_file.read(4), byteorder="big")
        num_of_rows = int.from_bytes(images_file.read(4), byteorder="big")
        num_of_colums = int.from_bytes(images_file.read(4), byteorder="big")
        labels_file.read(8)

        num_of_image_values = num_of_rows * num_of_colums
        data = [[None for x in range(num_of_image_values)]
                for y in range(num_of_items)]
        labels = []
        results = []
        i = 0
        for item in range(num_of_items):

            if n != -1:
                if item == n:
                    break

            print("Current image number: %7d" % item)
            for value in range(num_of_image_values):
                data[item][value] = int.from_bytes(images_file.read(1),
                                                   byteorder="big")
            # if edges == True:
            #     val = filters.sobel(np.array(data[item]).reshape(28, 28))
            #     m = max(val.reshape(28 * 28, ))
            #     val = (val / m) * 255.5
            #     data[item] = val.reshape(28 * 28, ).tolist()

            results.append((int.from_bytes(labels_file.read(1), byteorder="big"), data[item]))

        return results
    finally:
        images_file.close()
        labels_file.close()

def get_last_network(RES_FOLDER):
    p = re.compile('network[0-9]+')

    onlyfiles = [f for f in listdir(RES_FOLDER) if isfile(join(RES_FOLDER, f)) and p.fullmatch(f) is not None]
    if len(onlyfiles) == 0:
        print("nothing to evaluate" + time.ctime())
        return None
    res_dict = {}
    for f in onlyfiles:
        print(f)
        nbr = re.compile('[0-9]+').search(f).group(0)
        print(nbr)
        res_dict[int(nbr)] = f
    sorted_files_tuple = sorted(res_dict.items())
    print(sorted_files_tuple)
    return sorted_files_tuple[-1][1]

def caculate_sparsity(folder, batch_name=None):
    SOURCE_DATA = folder
    sparsity_batch = {}
    batch_name = batch_name if batch_name is not None else '_iden'
    spiking_batch = load_batch(batch_name, SOURCE_DATA).get('spike_trains')
    print(spiking_batch)
    sparsity_batch['sparsity_hist'] = get_sparsity(spiking_batch)
    save_batch(sparsity_batch, 'sparsity_batch', SOURCE_DATA)
    return sparsity_batch


def get_sparsity(spiking_batch):
    times = []
    for key in spiking_batch:
        times += [float(k) for k in spiking_batch[key] / ms]
    times = sorted(times)

    sparisty_hist = []

    for idx, spike_time in enumerate(times, 0):
        print(float(idx / len(times)) * 100)
        coocurrence = 0
        if idx == len(times) - 1:
            break
        for index_after_t in range(idx + 1, len(times)):
            if times[index_after_t] - spike_time <= 3:
                coocurrence += 1
            else:
                break
        sparisty_hist.append(coocurrence)
    return sparisty_hist

def get_neuron_img_mapping(results):
    neuron_img_map = {}
    for img_idx, firing in enumerate(results, 0):
        for neuron_idx, val in enumerate(firing, 0):
            if val != 0:
                current_val = neuron_img_map.get(neuron_idx, [])
                neuron_img_map[neuron_idx] = current_val + [img_idx]
    return neuron_img_map
