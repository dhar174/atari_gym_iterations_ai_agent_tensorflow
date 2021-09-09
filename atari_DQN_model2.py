import gym
import tensorflow as tf
##from tensorflow import keras
import keras
import random
import numpy as np
import datetime as dt
import imageio
import keyboard
import os
from tensorflow.keras import mixed_precision as mixed_precision
import tensorflow_model_optimization as tfmot
import tensorflow_addons as tfa

from keras.callbacks import History

from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard



es = EarlyStopping(monitor='loss', min_delta=1e-10, patience=10, verbose=1)
rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1)
mcp = ModelCheckpoint(filepath='weights.h5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True)

tb = TensorBoard('logs')



##history = History()
prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude


os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'
STORE_PATH = 'C:\\TensorFlowBook\\TensorBoard'
MAX_EPSILON = .4
MIN_EPSILON = 0.015
EPSILON_MIN_ITER = 500000
GAMMA = 0.96
BATCH_SIZE = 16
TAU = 0.005
POST_PROCESS_IMAGE_SIZE = (104, 88, 1)
DELAY_TRAINING = 512
NUM_FRAMES = 4
GIF_RECORDING_FREQ = 100
MODEL_SAVE_FREQ = 100
temp = 3.0
hidden_size = 32


new_policy = mixed_precision.experimental.Policy('mixed_float16')
print(new_policy.loss_scale)
mixed_precision.experimental.set_policy(new_policy)
print(tf.config.experimental.list_physical_devices('GPU'))
print('Compute dtype: %s' % new_policy.compute_dtype)
print('Variable dtype: %s' % new_policy.variable_dtype)
def image_preprocess(image, new_size=(104, 88)):
    # convert to greyscale, resize and normalize the image
    image = tf.image.rgb_to_grayscale(image)
    image = tf.image.resize(image, new_size)
    image = image / 255
    return image


env = gym.make("Phoenix-v0")
num_actions = env.action_space.n
##pruning_params = {
##      'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.50,
##                                                               final_sparsity=0.80,
##                                                               begin_step=40000,
##                                                               end_step=end_step)}
pruning_params = {
    'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(0.5, 10000, -1,1000),
    'block_size': (1, 1),
    'block_pooling_type': 'AVG'
}
def write_log(callback, names, logs, batch_no):
    writer = tf.summary.create_file_writer("C:/tmp/mylogs/eager")
    with writer.as_default():
        tf.summary.scalar("my_metric", 0.5, step=batch_no)
        writer.flush()

class DQNModel(keras.Model):
    def __init__(self, hidden_size: int, num_actions: int, dueling: bool):
        super(DQNModel, self).__init__()
        self.dueling = dueling
        self.conv1 = keras.layers.Conv2D(16, (8, 8), (4, 4), activation='relu',use_bias=False)
        self.conv2 = keras.layers.Conv2D(32, (4, 4), (2, 2), activation='relu',use_bias=False)
        self.conv3 = keras.layers.Conv2D(32, (4, 4), (2, 2), activation='relu',use_bias=True)
        self.dropout_layer = keras.layers.Dropout(rate=0.2)

##        self.pooling_layer1 = tf.keras.layers.MaxPool2D()
##        self.pooling_layer2 = tf.keras.layers.MaxPool2D()
##        self.pooling_layer3 = tf.keras.layers.MaxPool2D(1)
        self.flatten = keras.layers.Flatten()
        self.flatten1 = keras.layers.Flatten()
        self.adv_dense1 = keras.layers.Dense(hidden_size, activation='relu',use_bias=False,
                                         kernel_initializer=keras.initializers.he_normal())
        self.noise_layer = keras.layers.GaussianNoise(0.1)
        self.adv_dense2 = keras.layers.Dense(hidden_size, activation='relu',use_bias=False,
                                         kernel_initializer=keras.initializers.he_normal())
        self.adv_out = keras.layers.Dense(num_actions,activation='softmax',dtype=mixed_precision.Policy('float32'),
                                          kernel_initializer=keras.initializers.he_normal())
        if dueling:
            self.v_dense = keras.layers.Dense(hidden_size, activation='relu',use_bias=False,
                                         kernel_initializer=keras.initializers.he_normal())
            self.v_out = keras.layers.Dense(1, activation='softmax',dtype=mixed_precision.Policy('float32'),kernel_initializer=keras.initializers.he_normal())
            self.lambda_layer = keras.layers.Lambda(lambda x: x - tf.reduce_mean(x))
            self.combine = keras.layers.Add()

    def call(self, input):
        x = self.conv1(input)
        x = self.noise_layer(x)
##        x = self.pooling_layer1(x)
        x = self.dropout_layer(x)
        #x = self.flatten1(x)
        #x = self.adv_dense1(x)
##        print(x.shape)
        
        x = self.dropout_layer(x)
        x = self.conv2(x)
##        x = self.pooling_layer2(x)
##        print(x.shape)
        x = self.dropout_layer(x)
        x = self.conv3(x)
##        x = self.pooling_layer3(x)
        x = self.dropout_layer(x)
##        print(x.shape)
        x = self.flatten(x)
##        print(x.shape)
        adv = self.adv_dense2(x)
##        print(adv.shape)
        adv = self.dropout_layer(adv)
##        print("out",adv.shape)
        adv = self.adv_out(adv)
        
        if self.dueling:
            v = self.v_dense(x)
            v = self.v_out(v)
##            print('x(v)',v.shape)
            norm_adv = self.lambda_layer(adv)
##            print(norm_adv.shape)
            combined = self.combine([v, norm_adv])
            return combined
        return adv

primary_network = DQNModel(hidden_size, num_actions, True)
target_network = DQNModel(hidden_size, num_actions, True)


# make target_network = primary_network
for t, e in zip(target_network.trainable_variables, primary_network.trainable_variables):
    t.assign(e)
optimizer=mixed_precision.experimental.LossScaleOptimizer(keras.optimizers.Adamax(learning_rate=0.0000625),loss_scale='dynamic')
##optimizer = tf.train.experimental.enable_mixed_precision_graph_rewrite(optimizer)

#compile models
primary_network.compile(optimizer=optimizer, loss=tf.keras.losses.Huber(),metrics=['accuracy'])
target_network.compile(optimizer=optimizer, loss=tf.keras.losses.Huber(),metrics=['accuracy'])

loss_object = tf.keras.losses.Huber()

class Memory:
    def __init__(self, max_memory):
        self._max_memory = max_memory
        self._actions = np.zeros(max_memory, dtype=np.int32)
        self._rewards = np.zeros(max_memory, dtype=np.float32)
        self._frames = np.zeros((POST_PROCESS_IMAGE_SIZE[0], POST_PROCESS_IMAGE_SIZE[1], max_memory), dtype=np.float32)
        self._terminal = np.zeros(max_memory, dtype=np.bool)
        self._i = 0

    def add_sample(self, frame, action, reward, terminal):
        self._actions[self._i] = action
        self._rewards[self._i] = reward
        self._frames[:, :, self._i] = frame[:, :, 0]
        self._terminal[self._i] = terminal
        if self._i % (self._max_memory - 1) == 0 and self._i != 0:
            self._i = BATCH_SIZE + NUM_FRAMES + 1
        else:
            self._i += 1

    

    
    def sample(self):
        if self._i < BATCH_SIZE + NUM_FRAMES + 1:
##            print(self._i)
            raise ValueError("Not enough memory to extract a batch")
        else:
            with tf.device('/CPU:0'):
                rand_idxs = np.random.randint(NUM_FRAMES + 1, self._i, size=BATCH_SIZE)
                states = np.zeros((BATCH_SIZE, POST_PROCESS_IMAGE_SIZE[0], POST_PROCESS_IMAGE_SIZE[1], NUM_FRAMES),
                                 dtype=np.float32)
                next_states = np.zeros((BATCH_SIZE, POST_PROCESS_IMAGE_SIZE[0], POST_PROCESS_IMAGE_SIZE[1], NUM_FRAMES),
                                 dtype=np.float32)
                for i, idx in enumerate(rand_idxs):
                        states[i] = self._frames[:, :, idx - 1 - NUM_FRAMES:idx - 1]
                        next_states[i] = self._frames[:, :, idx - NUM_FRAMES:idx]
            return states, self._actions[rand_idxs], self._rewards[rand_idxs], next_states, self._terminal[rand_idxs]
    def clear(self):
        self._frames = []
        self._i = 0
memory = Memory(10000)
def policy(state, t):
    with tf.device('/CPU:0'):
        p = np.array([q[(state,x)]/t for x in range(env.action_space.n)])
        prob_actions = np.exp(p) / np.sum(np.exp(p))
        cumulative_probability = 0.0
        choice = random.uniform(0,1)
        for a,pr in enumerate(prob_actions):
            cumulative_probability += pr
            if cumulative_probability > choice:
                return a

def choose_action(state, primary_network, eps, step):
    if step < DELAY_TRAINING:
        return random.randint(0, num_actions - 1)
    else:
        if random.random() < eps:
            return random.randint(0, num_actions - 1)
        else:
##            print('action shape',tf.reshape(state, (1, POST_PROCESS_IMAGE_SIZE[0],
##                                                                POST_PROCESS_IMAGE_SIZE[1], NUM_FRAMES)).numpy().shape)
            return np.argmax(primary_network(tf.reshape(state, (1, POST_PROCESS_IMAGE_SIZE[0],
                                                           POST_PROCESS_IMAGE_SIZE[1], NUM_FRAMES)).numpy()))



def update_network(primary_network, target_network):
    # update target network parameters slowly from primary network
    for t, e in zip(target_network.trainable_variables, primary_network.trainable_variables):
        t.assign(t * (1 - TAU) + e * TAU)

def process_state_stack(state_stack, state):
##    print(state.shape)
    with tf.device('/CPU:0'):
        for i in range(1, state_stack.shape[-1]):
            state_stack[:, :, i - 1].assign(state_stack[:, :, i])
        state_stack[:, :, -1].assign(state[:, :, 0])
    return state_stack


def train(primary_network, memory, target_network=None):
    with tf.device('/CPU:0'):
        states, actions, rewards, next_states, terminal = memory.sample()
    # predict Q(s,a) given the batch of states
    prim_qt = primary_network(states)
    # predict Q(s',a') from the evaluation network
    prim_qtp1 = primary_network(next_states)
    # copy the prim_qt tensor into the target_q tensor - we then will update one index corresponding to the max action
    with tf.device('/CPU:0'):
        target_q = prim_qt.numpy()
        updates = rewards
        valid_idxs = terminal != True
        batch_idxs = np.arange(BATCH_SIZE)
    if target_network is None:
        with tf.device('/CPU:0'):
            updates[valid_idxs] += GAMMA * np.amax(prim_qtp1.numpy()[valid_idxs, :], axis=1)
    else:
        with tf.device('/CPU:0'):
            prim_action_tp1 = np.argmax(prim_qtp1.numpy(), axis=1)
            q_from_target = target_network(next_states)
        with tf.device('/CPU:0'):
            updates[valid_idxs] += GAMMA * q_from_target.numpy()[batch_idxs[valid_idxs], prim_action_tp1[valid_idxs]]
    with tf.device('/CPU:0'):
        target_q[batch_idxs, actions] = updates
        target_q = tf.Variable(tf.convert_to_tensor(target_q))
##    print('target-type ',type(target_q))
    states = np.asarray(states)
    target_q = np.asarray(target_q)
    callbacks=[es, rlr, mcp, tb]

    loss = primary_network.fit(states, target_q,batch_size=BATCH_SIZE,callbacks=callbacks)
    print(loss)
    loss = loss
    history = loss.history
##    loss = tf.Variable(tf.convert_to_tensor(loss))
##    print('loss-type ',type(loss))
    return loss

def evalu(primary_network, memory, target_network=None):
    with tf.device('/CPU:0'):
        states, actions, rewards, next_states, terminal = memory.sample()
    # predict Q(s,a) given the batch of states
    prim_qt = primary_network(states)
    # predict Q(s',a') from the evaluation network
    prim_qtp1 = primary_network(next_states)
    # copy the prim_qt tensor into the target_q tensor - we then will update one index corresponding to the max action
    with tf.device('/CPU:0'):
        target_q = prim_qt.numpy()
        updates = rewards
        valid_idxs = terminal != True
        batch_idxs = np.arange(BATCH_SIZE)
    if target_network is None:
        with tf.device('/CPU:0'):
            updates[valid_idxs] += GAMMA * np.amax(prim_qtp1.numpy()[valid_idxs, :], axis=1)
    else:
        with tf.device('/CPU:0'):
            prim_action_tp1 = np.argmax(prim_qtp1.numpy(), axis=1)
            q_from_target = target_network(next_states)
        with tf.device('/CPU:0'):
            updates[valid_idxs] += GAMMA * q_from_target.numpy()[batch_idxs[valid_idxs], prim_action_tp1[valid_idxs]]
    with tf.device('/CPU:0'):
        target_q[batch_idxs, actions] = updates
        target_q = tf.convert_to_tensor(target_q)
    callbacks=[es, rlr, mcp, tb]
    val_loss = primary_network.evaluate(tf.convert_to_tensor(states), target_q,batch_size=BATCH_SIZE,callbacks=callbacks)
    print(val_loss)
    val_loss = val_loss


    return val_loss


def named_logs(model, logs):
  result = {}
  for l in zip(model.metrics_names, logs):
    result[l[0]] = l[1]
  return result
def record_gif(frame_list, episode, fps=30):
    imageio.mimsave(STORE_PATH + f"/Pheonix-{episode}.gif", frame_list, fps=fps) #duration=duration_per_frame)

num_episodes = 1000000
eps = MAX_EPSILON
render = True
train_writer = tf.summary.create_file_writer(STORE_PATH + f"/DuelingQSI_{dt.datetime.now().strftime('%d%m%Y%H%M')}")
double_q = True
steps = 0
end_step = np.ceil(8 / BATCH_SIZE).astype(np.int32) * num_episodes


for i in range(num_episodes):
    state = env.reset()
    state = image_preprocess(state)
    with tf.device('/CPU:0'):
        state_stack = tf.Variable(np.repeat(state.numpy(), NUM_FRAMES).reshape((POST_PROCESS_IMAGE_SIZE[0],
                                                                                POST_PROCESS_IMAGE_SIZE[1],
                                                                                NUM_FRAMES)))


    
    
    cnt = 1
##    avg_loss = tf.Variable([0.0])
    avg_loss = 0
    #avg_loss.assign_add(0)
    tot_reward = 0
    if i % GIF_RECORDING_FREQ == 0:
        frame_list = []
    while True:
        if render:
            with tf.device('/GPU:0'):
                env.render()
##        print(state_stack.shape)
        with tf.device('/CPU:0'):
            action = choose_action(state_stack, primary_network, eps, steps)
        with tf.device('/GPU:0'):
            next_state, reward, done, info = env.step(action)
        tot_reward += reward
        if i % GIF_RECORDING_FREQ == 0:
            frame_list.append(tf.cast(tf.image.resize(next_state, (480, 320)), tf.uint8).numpy())
        next_state = image_preprocess(next_state)
        state_stack = process_state_stack(state_stack, next_state)
        # store in memory
        with tf.device('/CPU:0'):
            memory.add_sample(next_state, action, reward, done)

        if steps > DELAY_TRAINING:
            loss = train(primary_network, memory, target_network if double_q else None)
            history = loss.history
            val_loss = evalu(primary_network, memory, target_network if double_q else None)
            print(primary_network.metrics_names)
            update_network(primary_network, target_network)
            callbacks=[es, rlr, mcp, tb]
            val_names = ['loss', 'accuracy']
            print(type(loss.history))
            print(type(loss))
            print(type(cnt))

            
            for call in callbacks:
                call.set_model(primary_network)
                call.on_epoch_end(cnt, named_logs(primary_network, loss.history))
            write_log(callbacks, val_names, loss, i//10)
            loss = loss[0]
##            print(history.history.keys())
##            print('loss ',loss)
##            print(loss_object[0])
##            loss_object = tf.Variable(loss[-1])
##            cur_loss = tf.reshape(loss_object, [1,])
##            print('avg_loss shape ',avg_loss.shape)
##            print('cur_loss ',cur_loss)
##            print('loss object numpy',loss_object.numpy())
##            avg_loss.assign_add(cur_loss)
##            print(type(avg_loss))
##            print(type(cur_loss))
##            print("Loss: ",primary_network.metrics[0].total.value)
##            print("Acc: ",primary_network.metrics[1])
            
            ##print(tot_reward)
           


        else:
            loss = -1
        
        

        # linearly decay the eps value
        if steps > DELAY_TRAINING:
            with tf.device('/CPU:0'):
                eps = MAX_EPSILON - ((steps - DELAY_TRAINING) / EPSILON_MIN_ITER) * \
                      (MAX_EPSILON - MIN_EPSILON) if steps < EPSILON_MIN_ITER else \
                    MIN_EPSILON
        steps += 1

        if done:
            if steps > DELAY_TRAINING:
                avg_loss = (avg_loss / cnt)
##                if(tot_reward > 100):
##            
##                    with tf.device('/GPU:0'):
##                        input_shape= (num_actions,104,88,4)
##                        new_model = tf.keras.Sequential()
##                        new_model.add(keras.layers.Conv2D(16, (8, 8), (4, 4), activation='relu',use_bias=False,input_shape= (num_actions,104,88,4),name="1"))
##                        new_model.add(keras.layers.Conv2D(32, (4, 4), (2, 2), activation='relu',use_bias=False,name="2"))
##                        new_model.add(keras.layers.Conv2D(32, (4, 4), (2, 2), activation='relu',use_bias=False,name="3"))
##                        new_model.add(keras.layers.Dropout(rate=0.2)name="4")
##                        new_model.add(keras.layers.Flatten(),name="5")
##                        new_model.add(keras.layers.Flatten(),name="6")
##                        new_model.add(keras.layers.Dense(hidden_size, activation='relu',use_bias=False,
##                                         kernel_initializer=keras.initializers.he_normal(),name="7"))
##                        
##                        new_model.add(keras.layers.GaussianNoise(0.1),name="8")
##                        new_model.add(keras.layers.Dense(hidden_size, activation='relu',use_bias=False,
##                                         kernel_initializer=keras.initializers.he_normal(),name="9"))
##                        new_model.add(keras.layers.Dense(num_actions,activation='softmax',dtype=mixed_precision.Policy('float32'),
##                                                         kernel_initializer=keras.initializers.he_normal(),name="10"))
##                        new_model.add(keras.layers.Dense(hidden_size, activation='relu',use_bias=False,
##                                         kernel_initializer=keras.initializers.he_normal(),name="11"))
##                        new_model.add(keras.layers.Dense(1, activation='softmax',dtype=mixed_precision.Policy('float32'),kernel_initializer=keras.initializers.he_normal()))
##                        new_model.add(keras.layers.Lambda(lambda x: x - tf.reduce_mean(x),name="12"))
##                        new_model.add(keras.layers.add([new_model.get_layer('')]))
##                      
##                        new_model.build(input_shape)
##                        primary_network.build(input_shape)
##                        target_network.build(input_shape)
##                        print(primary_network.variables)
##                        print(new_model.variables)
##                        for a, b in zip(new_model.variables, primary_network.variables):
##                            a.assign(b)
##                        for a, b in zip(new_model.get_weights, primary_network.get_weights):
##                            a.set_weights(b)
##                        model_for_pruning = prune_low_magnitude(new_model, **pruning_params)
##                        
##                        model_for_pruning.compile(optimizer= optimizer,
##                          loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
##                                                  metrics=['accuracy'])
##                        for a, b in zip(primary_network.variables, new_model.variables):
##                            a.assign(b)
##                        for a, b in zip(primary_network.get_weights, new_model.get_weights):
##                            a.set_weights(b)
##                        primary_network = model_for_pruning
                print(type(loss))
                print(type(tot_reward))
                print(type(eps))
                print(f"Episode: {i}, Reward: {tot_reward}, avg loss: {loss:.5f}, eps: {eps:.3f}")
                with train_writer.as_default():
                    tf.summary.scalar('reward', tot_reward, step=i)
                    tf.summary.scalar('avg loss', avg_loss, step=i)
            else:
                print(f"Pre-training...Episode: {i}")
            if i % MODEL_SAVE_FREQ == 0:
                primary_network.save_weights(STORE_PATH+f"/KungFuMaster-{i}/", overwrite=True)

            if i % GIF_RECORDING_FREQ == 0:
                record_gif(frame_list, i)
            break

        cnt += 1
        if keyboard.is_pressed('c') and steps > DELAY_TRAINING:
            primary_network.save_weights(STORE_PATH+f"/KungFuMaster-{i}/", overwrite=True)
    if i % MODEL_SAVE_FREQ == 0: # and i != 0:
        primary_network.save_weights(STORE_PATH + "/checkpoints2/cp_primary_network_episode_{}.ckpt".format(i), overwrite=True)
        target_network.save_weights(STORE_PATH + "/checkpoints2/cp_target_network_episode_{}.ckpt".format(i), overwrite=True)
        try:
            param = primary_network.count_params()
            print(param)
            input_shape= (num_actions,104,88,4)
            #config = primary_network.get_config()
    ##        primary_network.compile(optimizer=keras.optimizers.Adam(), loss=tf.keras.losses.Huber())
    ##        target_network.compile(optimizer=keras.optimizers.Adam(), loss=tf.keras.losses.Huber())
            primary_network.build(input_shape)
            target_network.build(input_shape)
            primary_network.save(STORE_PATH + "/checkpoints2/cp_primary_network", save_format='tf',overwrite=True)
            target_network.save(STORE_PATH + "/checkpoints2/cp_target_network", save_format='tf',overwrite=True)
            print("Saved!")
        except:
            print("Failed to save :-(")
env.close()
