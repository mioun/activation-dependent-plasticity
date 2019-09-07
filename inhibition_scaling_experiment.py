from SSNNetwork import *
import sys

net1 = load_batch('network2', DATA_PATH + '/' + 'Simple_0.2_100_')

argv = sys.argv

# inhibition weight w_inh
inh = int(sys.argv[1])

# scaling constant taur
taur = int(sys.argv[2])

RES_FOLDER = BATCH_FOLDER + '/' + 'inhibition_scaling_experiment_' + str(taur) + "_" + str(inh)

set_device('cpp_standalone')


def test(RES_FOLDER):
    N = 100
    if not os.path.exists(BATCH_FOLDER + '/' + '6000_1batch_'):
        generate_trianing_batch(6000, 1)
    training_batch = load_batch('6000_1batch_', BATCH_FOLDER)
    network_batch = None

    for i in range(0, 4):
        net = NetworkBuilder('_train', N, init_batch=network_batch, taur=taur * ms, inh=inh)
        network_batch = train(training_batch, net)
        save_batch(network_batch, 'network' + str(i), RES_FOLDER)

    net = NetworkBuilder('_test', ext_size=N, init_batch=network_batch, inh=inh)
    iden_batch = identify_classes(training_batch, net)
    save_batch(iden_batch, '_iden', RES_FOLDER)

    if not os.path.exists(BATCH_FOLDER + '/' + 'test_batch'):
        generate_test_batch()
    test_batch = load_batch('test_batch', BATCH_FOLDER)

    net = NetworkBuilder('_test', ext_size=N, init_batch=network_batch, inh=8)
    results = net.simulate_for_results(test_batch)
    res_dict = {'results': results, 'labels': test_batch.get('labels')}
    save_batch(res_dict, '_eval', RES_FOLDER)

    res = evaluate_neurons(test_batch.get('labels'), results, iden_batch)

    print('scalar product', res.get('pred_vac'))
    print('highest  rate  classification', res.get('pred_neu'))

    save_batch(res, '_final_results', RES_FOLDER)


test(RES_FOLDER)
