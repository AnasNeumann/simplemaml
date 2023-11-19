# SIMPLE MAML
A generic Python and TensorFlow function that implements a simple version of the "Model-Agnostic Meta-Learning (MAML) Algorithm for Fast Adaptation of Deep Networks" as designed by Chelsea Finn et al. 2017 [1]. Especially, this implementation focuses on regression and prediction problems. 

## Original algorithm adapted for regression
![original-algorithm](/MAML.png)

## Usage
1. Install with `pip install simplemaml`
2. In your python code:
    - `from simplemaml import MAML`
    - `MAML(model=your_model, tasks=your_array_of_tasks, etc.)`
3. Your task should be in one of the two follwing formats:
    - `tasks=[{"inputs": [], "target": []}, etc.]`
    - `tasks=[{"train": {"inputs": [], "target": []}, "test": {"inputs": [], "target": []}}, etc.]`

## More about the algorithm
* Chelsea Finn explains well her algorithm in this Standford lecture: https://www.youtube.com/watch?v=Gj5SEpFIv8I&list=PLoROMvodv4rNjRoawgt72BBNwL2V7doGI
* Original repository with a more complete version of the code: https://github.com/cbfinn/maml

## Tools needed
* tensorflow>=2.13.0: https://www.tensorflow.org/
* numpy>=1.24.3: https://numpy.org/

## Refer to this Repository in scientific document
Neumann, Anas. (2023). Simple Python and TensorFlow implementation of the optimization-based Model-Agnostic Meta-Learning (MAML) algorithm for supervised regression problems. *GitHub repository: https://github.com/AnasNeumann/simplemaml*.

```bibtex
    @misc{simplemaml,
      author = {Anas Neumann},
      title = {Simple Python and TensorFlow implementation of the optimization-based Model-Agnostic Meta-Learning (MAML) algorithm for supervised regression problems},
      year = {2023},
      publisher = {GitHub},
      journal = {GitHub repository},
      howpublished = {\url{https://github.com/AnasNeumann/simplemaml}},
      commit = {main}
    }
```

## Complete code
```python
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
    if tf.config.list_physical_devices('GPU'):
        with tf.device('/GPU:0'):
            return _MAML_compute(model, alpha, beta, optimizer, c_loss, f_loss, meta_epochs, meta_tasks_per_epoch, inputs_dimension, validation_split, k_folds, tasks, cumul)
    else:
       return _MAML_compute(model, alpha, beta, optimizer, c_loss, f_loss, meta_epochs, meta_tasks_per_epoch, inputs_dimension, validation_split, k_folds, tasks, cumul)

def _build_task(t, inputs_dimension, validation_split, k_folds):
    """
    Build task t by splitting train_input, test_input, train_target, test_target if it's not already done.
    This function is flexible and handle both randon validation_splits and k_folds.
        :param t: a task to learn during the meta-pre-training stage
        :param inputs_dimension: the input dimension (for sequence-to-sequence models).
        :param validation_split: optional ratio of data to use for training in each task (could be fixed or random between two values).
        :param k_folds: optional cross-validation with k_folds each time a task is called for meta-learning.
        :return: train_input, test_input, train_target, test_target
    """
    if "train" in t and "test" in t:
        train_input = t["train"]["inputs"] if inputs_dimension<=1 else [t["train"]["inputs"] for d in range(inputs_dimension)]
        test_input = t["test"]["inputs"] if inputs_dimension<=1 else [t["test"]["inputs"] for d in range(inputs_dimension)]
        return t["train"]["inputs"], t["test"]["inputs"], t["train"]["target"], t["test"]["target"] 
    elif k_folds>0:
        fold = random.randint(0, k_folds-1)
        fold_size = (len(t["inputs"]) // k_folds)
        v_start = fold * fold_size
        v_end = (fold + 1) * fold_size if fold < k_folds - 1 else len(t["inputs"])
        t_i = np.concatenate((t["inputs"][:v_start], t["inputs"][v_end:]), axis=0)
        train_input = t_i if inputs_dimension<=1 else [t_i for d in range(inputs_dimension)]
        test_input = t["inputs"][v_start:v_end] if inputs_dimension<=1 else [t["inputs"][v_start:v_end] for d in range(inputs_dimension)]
        train_target = np.concatenate((t["target"][:v_start], t["target"][v_end:]), axis=0)
        test_target = t["target"][v_start:v_end]
        return train_input, test_input, train_target, test_target
    else:
        v = random.uniform(validation_split[0], validation_split[1]) if isinstance(validation_split,list) else validation_split
        split_idx = int(len(t["inputs"]) * v)
        train_input = t["inputs"][:split_idx] if inputs_dimension<=1 else [t["inputs"][:split_idx] for _ in range(inputs_dimension)]
        test_input = t["inputs"][split_idx:] if inputs_dimension<=1 else [t["inputs"][split_idx:] for _ in range(inputs_dimension)]
        train_target, test_target = t["target"][:split_idx], t["target"][split_idx:]
        return train_input, test_input, train_target, test_target

def _MAML_compute(model, alpha, beta, optimizer, c_loss, f_loss, meta_epochs, meta_tasks_per_epoch, inputs_dimension, validation_split, k_folds, tasks, cumul):
    log_step = meta_epochs // 10 if meta_epochs > 10 else 1
    optim_test=optimizer(learning_rate=alpha)
    optim_test.build(model.trainable_variables)
    model.compile(loss=f_loss, optimizer=optim_test)
    losses=[]
    total_loss=0.
    for step in range (meta_epochs):
        sum_gradients = [tf.zeros_like(variable) for variable in model.trainable_variables]
        num_tasks_sampled = random.randint(meta_tasks_per_epoch[0], meta_tasks_per_epoch[1])
        model_copy = tf.keras.models.clone_model(model)
        model_copy.build(model.input_shape)
        model_copy.set_weights(model.get_weights())
        optim_train=optimizer(learning_rate=beta)
        optim_train.build(model_copy.trainable_variables)
        model_copy.compile(loss=f_loss, optimizer=optim_train)
        for _ in range(num_tasks_sampled):
            gc.collect()
            t = random.choice(tasks)
            train_input, test_input, train_target, test_target = _build_task(t, inputs_dimension, validation_split, k_folds)
            
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
        gc.collect()
    return model, losses
```

## REFERENCES
[1] Finn, C., Abbeel, P. &amp; Levine, S.. (2017). Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks. <i>Proceedings of the 34th International Conference on Machine Learning</i>, in <i>Proceedings of Machine Learning Research</i> 70:1126-1135 Available from https://proceedings.mlr.press/v70/finn17a.html and https://proceedings.mlr.press/v70/finn17a/finn17a.pdf.
