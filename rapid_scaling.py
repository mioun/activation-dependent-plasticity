from SSNNetwork import *
import sys

argv = sys.argv
# rapid scaling Ro
Ro = float(sys.argv[1])
# excitatory time constant increase needed after rapid scaling
ext = float(sys.argv[2])
# name of the network for rapid scaling
start_batch_name = sys.argv[3]
# folder where networks are stored
start_folder_name = sys.argv[4]

RES_FOLDER = BATCH_FOLDER + '/' + start_folder_name

# set_device('cpp_standalone')


def test():
    if not os.path.exists(BATCH_FOLDER + '/' + '6000_1batch_'):
        generate_trianing_batch(6000, 1)
    training_batch = load_batch('6000_1batch_', BATCH_FOLDER)
    network_batch = load_batch(start_batch_name, RES_FOLDER)
    N = int(len(network_batch.get('weights')) / 784)
    print(N)
    suffix = start_batch_name + str(Ro)

    # training
    net = NetworkBuilder('_train', N, init_batch=network_batch, alpha=Ro, inh=8)
    network_batch = train(training_batch, net)
    save_batch(network_batch, 'network' + suffix, RES_FOLDER)

    percent_batch = training_batch

    n_suffix = suffix + str(ext)

    # evaluation
    net = NetworkBuilder('_test', ext_size=N, init_batch=network_batch, inh=8, ext=1 + ext)
    iden_batch = identify_classes(percent_batch, net)
    save_batch(iden_batch, '_iden' + n_suffix, RES_FOLDER)

    if not os.path.exists(BATCH_FOLDER + '/' + 'test_batch'):
        generate_test_batch()
    test_batch = load_batch('test_batch', BATCH_FOLDER)
    net = NetworkBuilder('_test', ext_size=N, init_batch=network_batch, inh=8, ext=1 + ext)
    results = net.simulate_for_results(test_batch)
    res_dict = {'results': results, 'labels': test_batch.get('labels')}
    save_batch(res_dict, '_eval' + n_suffix, RES_FOLDER)

    res = evaluate_neurons(test_batch.get('labels'), results, iden_batch)
    print('scalar product', res.get('pred_vec'))
    print('highest  rate  classification', res.get('pred_neu'))
    save_batch(res, '_final_results' + n_suffix, RES_FOLDER)


test()
