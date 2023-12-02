import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import random
import gc

def MAML(model, alpha=0.005, beta=0.005, optimizer=keras.optimizers.SGD, c_loss=keras.losses.mse, f_loss=keras.losses.MeanSquaredError(), meta_epochs=100, meta_tasks_per_epoch=[10, 30], inputs_dimension=1, validation_split=0.2, k_folds=0, tasks=[], cumul=False):
    """
    Simple MAML algorithm implementation for supervised regression.
        :param model: A Keras model to be trained using MAML.
        :param alpha: Learning rate for task-specific updates.
        :param beta: Learning rate for meta-updates.
        :param optimizer: Optimizer to be used for training.
        :param c_loss: Loss function for calculating training loss.
        :param meta_epochs: Number of meta-training epochs.
        :param meta_tasks_per_epoch: Range of tasks to sample per epoch.
        :param inputs_dimension: the input dimension (for sequence-to-sequence models).
        :param validation_split: Ratio of data to use for validation in each task (could be fixed or random between two values).
        :param k_folds: cross-validation with k_folds each time a task is called for meta-learning.
        :param tasks: List of tasks for meta-training.
        :param cumul: choose between sum and mean gradients during the outer loop.
        :return: Tuple of trained model and evolution of losses over epochs.
    """
    if "train" in tasks[0] and "test" in tasks[0]:
        build_task_f = _get_task
        build_task_param = {"dimension": inputs_dimension}
    elif k_folds>0:
        build_task_f = _k_fold_task
        build_task_param = {"dimension": inputs_dimension, "k": k_folds}
    else:
        build_task_f = _split_task
        build_task_param = {"dimension": inputs_dimension, "split": validation_split}
    if tf.config.list_physical_devices('GPU'):
        with tf.device('/GPU:0'):
            return _MAML_compute(model, alpha, beta, optimizer, c_loss, f_loss, meta_epochs, meta_tasks_per_epoch, build_task_f, build_task_param, tasks, cumul)
    else:
       return _MAML_compute(model, alpha, beta, optimizer, c_loss, f_loss, meta_epochs, meta_tasks_per_epoch, build_task_f, build_task_param, tasks, cumul)

def _split_task(t, param):
    d = param["dimension"]
    split = param["split"]
    v = random.uniform(split[0], split[1]) if isinstance(split,list) else split
    split_idx = int(len(t["inputs"]) * v)
    train_input = t["inputs"][:split_idx] if d<=1 else [t["inputs"][:split_idx] for _ in range(d)]
    test_input = t["inputs"][split_idx:] if d<=1 else [t["inputs"][split_idx:] for _ in range(d)]
    train_target, test_target = t["target"][:split_idx], t["target"][split_idx:]
    return train_input, test_input, train_target, test_target

def _k_fold_task(t, param):
    d = param["dimension"]
    k = param["k"]
    fold = random.randint(0, k-1)
    fold_size = (len(t["inputs"]) // k)
    v_start = fold * fold_size
    v_end = (fold + 1) * fold_size if fold < k - 1 else len(t["inputs"])
    t_i = np.concatenate((t["inputs"][:v_start], t["inputs"][v_end:]), axis=0)
    train_input = t_i if d<=1 else [t_i for d in range(d)]
    test_input = t["inputs"][v_start:v_end] if d<=1 else [t["inputs"][v_start:v_end] for _ in range(d)]
    train_target = np.concatenate((t["target"][:v_start], t["target"][v_end:]), axis=0)
    test_target = t["target"][v_start:v_end]
    return train_input, test_input, train_target, test_target

def _get_task(t, param):
    d = param["dimension"]
    train_input = t["train"]["inputs"] if d<=1 else [t["train"]["inputs"] for _ in range(d)]
    test_input = t["test"]["inputs"] if d<=1 else [t["test"]["inputs"] for _ in range(d)]
    return train_input, test_input, t["train"]["target"], t["test"]["target"] 

def _MAML_compute(model, alpha, beta, optimizer, c_loss, f_loss, meta_epochs, meta_tasks_per_epoch, build_task_f, build_task_param, tasks, cumul):
    log_step = meta_epochs // 10 if meta_epochs > 10 else 1
    optim_test=optimizer(learning_rate=alpha)
    optim_train=optimizer(learning_rate=beta)
    model_copy = tf.keras.models.clone_model(model)
    model_copy.build(model.input_shape)
    model_copy.set_weights(model.get_weights())
    optim_test.build(model.trainable_variables)
    optim_train.build(model_copy.trainable_variables)
    model.compile(loss=f_loss, optimizer=optim_test)
    model_copy.compile(loss=f_loss, optimizer=optim_train)
    losses=[]
    total_loss=0.
    for step in range (meta_epochs):
        sum_gradients = [tf.zeros_like(variable) for variable in model.trainable_variables]
        num_tasks_sampled = random.randint(meta_tasks_per_epoch[0], meta_tasks_per_epoch[1])
        model_copy.set_weights(model.get_weights())
        for _ in range(num_tasks_sampled):
            train_input, test_input, train_target, test_target = build_task_f(random.choice(tasks), build_task_param)

            # 1. Inner loop: Update the model copy on the current task
            with tf.GradientTape(watch_accessed_variables=False) as train_tape:
                train_tape.watch(model_copy.trainable_variables)
                train_pred = model_copy(train_input)
                train_loss = tf.reduce_mean(c_loss(train_target, train_pred))
            g = train_tape.gradient(train_loss, model_copy.trainable_variables)
            optim_train.apply_gradients(zip(g, model_copy.trainable_variables))

            # 2. Compute gradients with respect to the test data
            with tf.GradientTape(watch_accessed_variables=False) as test_tape:
                test_tape.watch(model_copy.trainable_variables)
                test_pred = model_copy(test_input)
                test_loss = tf.reduce_mean(c_loss(test_target, test_pred))
            g = test_tape.gradient(test_loss, model_copy.trainable_variables)
            for i, gradient in enumerate(g):
                sum_gradients[i] += gradient

        # 3. Meta-update: apply the accumulated gradients to the original model
        cumul_gradients = [grad / (1.0 if cumul else num_tasks_sampled) for grad in sum_gradients]
        optim_test.apply_gradients(zip(cumul_gradients, model.trainable_variables))
        total_loss += test_loss.numpy()
        loss_evol = total_loss/(step+1)
        losses.append(loss_evol)
        if step % log_step == 0:
            print(f'Meta epoch: {step+1}/{meta_epochs},  Loss: {loss_evol}')
    return model, losses
