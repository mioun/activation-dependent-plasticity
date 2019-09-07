from utils import *
import time

defaultclock.dt = 0.5 * ms

############### Synaptic equations definition ########################
stimulus_syn_equations = {}
stimulus_syn_equations['eq_train'] = ''' w : 1
                 taupre : second
                 w_max : 1
                 w_min : 1
                 eta : 1
                 alpha : 1
                 dApre/dt=-Apre/taupre : 1 (event-driven)
                 '''
stimulus_syn_equations['pre_train'] = '''
                 ge += w
                 Apre = 1
                 '''
stimulus_syn_equations['post_train'] = '''
                 w = clip(w + 0.01*eta*(Apre-(R+alpha)*w),w_min,w_max)
                 '''
stimulus_syn_equations['eq_test'] = '''
                 w : 1
                 '''
stimulus_syn_equations['pre_test'] = '''
                 ge += w
                 '''
stimulus_syn_equations['post_test'] = '''
                 '''

logger = get_logger()


class NetworkBuilder:

    def __init__(self, tribe, ext_size=100, init_batch=None, eta=1, alpha=0, inh=3, ext=1,
                 refractory= 3 * ms, taur=70 * ms):
        self.tribe = tribe
        self.stimulus_size = 28 * 28
        self.ext_size = ext_size
        self.voltage_mon_dict = {}
        self.init_batch = init_batch
        self.weights_monitor = {}
        self.learn_date_size = 0
        self.exposition_time = 0 * ms
        self.rest_time = 0 * ms
        self.conductance_mon = None
        self.epoch = 0
        self.M = None
        self.alpha = alpha
        self.inh = inh
        self.taur = taur
        self.ext = ext
        self.eta = eta
        self.spike_mon = None
        self.inh_size = 1
        self.w_max = 1
        self.w_min = 0
        self.init_weights = generate_random_weights(self.stimulus_size, self.ext_size, self.w_max)
        self.stimulus_group = SpikeGeneratorGroup(self.stimulus_size, [0], [1] * ms)
        self.ext_group = self.build_ext()
        self.inh_group = self.build_inh()
        self.stim_ext_syn = self.stim_ext_synapses()
        self.ext_inh_syn = self.ext_inh_synapses()
        self.inh_ext_syn = self.inh_ext_synapses()

        self.network = Network(self.stimulus_group,
                               self.ext_group,
                               self.stim_ext_syn,
                               self.inh_group,
                               self.inh_ext_syn,
                               self.ext_inh_syn
                               )

    def build_ext(self):
        ext_eq = '''
            dv/dt = (Ie+(vr-v)+Inh)/ taum : volt (unless refractory)
            Ie = ge * (Ee-v): volt 
            Inh = ginh*(Einh-v) : volt
            dge/dt = -ge / taue : 1 
            dginh /dt = -ginh  / (taui): 1 
            dR/dt = -R/taur : 1
            taum : second
            taue : second
            taur : second
            taui: second     
            vr : volt
            vt : volt
            Ee : volt
            Einh :volt
            '''

        vt_eq = 'v > vt'

        group = NeuronGroup(self.ext_size, ext_eq, reset=' R += 1; v = vr', method='euler',
                            threshold=vt_eq, refractory='3*ms', name='ext')
        group.v = -65 * mV
        group.vr = -65 * mV
        group.vt = -52 * mV
        group.taum = 100 * ms
        group.taue = self.ext * ms
        group.taui = 2 * ms
        group.taur = self.taur
        group.Einh = -90 * mV
        group.Ee = 0 * mV
        return group

    def build_inh(self):
        inh_eq = '''
            dv/dt = (ge * (Ee-v)+ vr-v)/ taum : volt (unless refractory)
            dge/dt = -ge / taui: 1 (unless refractory)
            taum : second
            taui : second
            vr : volt
            Ee : volt
            vt : volt
            '''
        group = NeuronGroup(self.inh_size, inh_eq,
                            reset='v = vr;', method='euler',
                            threshold='v > vt', refractory='1*ms')
        group.v = -65 * mV
        group.vr = -65 * mV
        group.vt = -50 * mV
        group.taum = 20 * ms
        group.taui = 20 * ms
        group.Ee = 0 * mV
        return group

    def stim_ext_synapses(self):
        S = Synapses(self.stimulus_group, self.ext_group,
                     stimulus_syn_equations['eq' + self.tribe],
                     on_pre=stimulus_syn_equations['pre' + self.tribe],
                     on_post=stimulus_syn_equations['post' + self.tribe], method='euler', )
        S.connect()
        if self.tribe == '_test':
            if self.init_batch is not None:
                print("loading weights from a batch" + str(len(self.init_batch.get('weights'))))
                if len(self.init_batch.get('weights')) != self.ext_size * self.stimulus_size:
                    raise ValueError('weights in batch not match network size ')
                self.load_weights_delays(self.init_batch.get('weights'), self.init_batch.get('delays'), S)
            return S
        initial_values = {
            'w_max': self.w_max,
            'w_min': self.w_min,
            'taupre': 5 * ms,
            'eta': self.eta,
            'alpha': self.alpha}
        if self.init_batch is not None:
            self.load_weights_delays(self.init_batch.get('weights'), self.init_batch.get('delays'), S)
        else:
            S.w = self.init_weights.flatten()
            S.delay = np.random.rand(self.stimulus_size, self.ext_size).flatten() * 15 * ms
        S.set_states(initial_values)
        return S

    def ext_inh_synapses(self):
        S = Synapses(self.ext_group, self.inh_group,
                     '''wi : 1
                 ''',
                     on_pre='''
                     ge += wi
                 '''
                     )
        S.connect()
        S.wi = self.w_max * 5
        return S

    def inh_ext_synapses(self):
        S = Synapses(self.inh_group, self.ext_group,
                     ''' w_inh : 1
                         taui_i : second
                         w_max : 1
                         w_min : 1
                 ''',
                     on_pre='''
                     ginh += w_inh
                 ''')

        S.connect()
        S.taui_i = 10 * ms
        S.w_max = self.w_max
        S.w_min = self.w_min
        S.w_inh = self.w_max * self.inh
        return S

    def run(self, runtime):
        self.network.run(runtime, report='text')

    def learn(self, batch):
        self.stimulus_group.set_spikes(batch.get('indicies'), batch.get('times'))  # array(elem[1])*Hz
        self.ext_group.ginh = 0
        defaultclock.dt = 0.5 * ms
        self.run(batch.get('sim_time'))

    def read_weight_delay(self):
        print('reading')
        weights = self.stim_ext_syn.get_states()['w']
        delays = self.stim_ext_syn.get_states()['delay']
        return (weights, delays)

    def register_spike_monitor(self):
        S_M = SpikeMonitor(self.ext_group, record=True, name="SM")
        self.spike_mon = S_M
        self.network.add(S_M)

    def save_weight_delays(self):
        weights = self.stim_ext_syn.get_states()['w']
        delays = self.stim_ext_syn.delay
        train_batch = {}
        train_batch['weights'] = weights
        train_batch['delays'] = array(delays)
        return train_batch

    def load_weights_delays(self, weights_list, delays_list, synapses):
        synapses.w = weights_list
        synapses.delay = delays_list * ms

    def simulate_for_results(self, batch, save_spikes=False):
        self.register_spike_monitor()
        self.stimulus_group.set_spikes(batch.get('indicies'), batch.get('times'))
        self.ext_group.ginh = 0
        defaultclock.dt = 0.5 * ms
        self.run(batch.get('sim_time'))
        defaultclock.dt = 0.5 * ms
        return spike_result_array(batch.get('exposition_time'),
                                  batch.get('rest_time'),
                                  batch.get('N'),
                                  self.spike_mon.spike_trains(),
                                  batch.get('start_time'))

    def identify_classes(self, labeled_batch):
        results = self.simulate_for_results(labeled_batch)
        device.reinit()
        device.activate()
        labels = labeled_batch.get('labels')
        firing_for_label_dict = {}
        for label in labels:
            firing_for_label_dict[label] = [0] * self.ext_size
        for image_idx, label in enumerate(labels, 0):
            firing_for_label_dict[label] = add(firing_for_label_dict.get(label), results.get('results')[image_idx])
        identify_batch = {'results': results.get('results'),
                          'labels': labels,
                          'firing_for_label_dict': firing_for_label_dict,
                          'neuron_class_mapping': map_neuron_to_class(firing_for_label_dict),
                          'spike_trains': results.get('spike_trains')
                          }
        return identify_batch


def train(batch, net):
    t1 = time.clock()
    print("building network: " + str(t1 - time.clock()))
    net.ext_inh_syn.active = True
    net.inh_ext_syn.active = True
    net.learn(batch)
    net_batch = net.save_weight_delays()
    device.reinit()
    device.activate()
    return net_batch


def identify_classes(labeled_batch, net):
    net.ext_inh_syn.active = True
    net.inh_ext_syn.active = True
    return net.identify_classes(labeled_batch)


def test(net, evaluate_batch, identify_batch):
    results = net.simulate_for_results(evaluate_batch)
    return evaluate_neurons(evaluate_batch, results, identify_batch)
