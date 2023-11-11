# MAML
A generic Python/Tensorflow function that implements a simple version of the "Model-Agnostic Meta-Learning (MAML) Algorithm for Fast Adaptation of Deep Networks" as designed by Chelsea Finn et al. 2017 [1]. Especially, this implementation focuses on regression and prediction problems. 

## More about the algorithm
* Chelsea Finn explains well her algorithm in this Standford lecture: https://www.youtube.com/watch?v=Gj5SEpFIv8I&list=PLoROMvodv4rNjRoawgt72BBNwL2V7doGI
* Original repository with a more complete version of the code: https://github.com/cbfinn/maml

## Original algorithm adapted for regression
![original-algorithm](/MAML.png)

## Usage
1. Install with `pip install simplemaml`
2. In your python code:
    - `import simplemaml.MAML as MAML`
    - `MAML(model=your_model, tasks=your_array_of_tasks, etc.)`

## Tools needed
* tensorflow>=2.13.0: https://www.tensorflow.org/
* numpy>=1.24.3: https://numpy.org/

## Complete code
```python
# MAML generic function
def MAML(model, alpha=0.005, beta=0.005, optimizer=keras.optimizers.Adam, c_loss=keras.losses.mse, f_loss=keras.losses.MeanSquaredError(), meta_epochs=100, meta_tasks_per_epoch=[10, 30],   train_split=0.2, tasks=[]):
    log_step = meta_epochs // 10 if meta_epochs > 10 else 1
    optim_test=optimizer(learning_rate=alpha)
    optim_test.build(model.trainable_variables)
    model.compile(loss=f_loss, optimizer=optim_test)
    losses=[]
    total_l=0.
    for step in range (meta_epochs):
        task_gradients = []
        model_copy = tf.keras.models.clone_model(model)
        model_copy.build(model.input_shape)
        model_copy.set_weights(model.get_weights())
        optim_train=optimizer(learning_rate=beta)
        optim_train.build(model_copy.trainable_variables)
        model_copy.compile(loss=f_loss, optimizer=optim_train)
        for _ in range(random.randint(meta_tasks_per_epoch[0], meta_tasks_per_epoch[1])):
            t = tasks[random.randint(0, len(tasks)-1)]
            split_idx = int(len(t["inputs"]) * train_split)
            train_input  = t["inputs"][:split_idx]
            test_input = t["inputs"][split_idx:]
            train_target  = t["target"][:split_idx]
            test_target = t["target"][split_idx:]
            # Inner loop: Update the model copy on the current task
            with tf.GradientTape(watch_accessed_variables=False) as train_tape:
                train_tape.watch(model_copy.trainable_variables)
                train_pred = model_copy(train_input)
                train_loss = tf.reduce_mean(c_loss(train_target, train_pred))
            g = train_tape.gradient(train_loss, model_copy.trainable_variables)
            optim_train.apply_gradients(zip(g, model_copy.trainable_variables))
            # Compute gradients with respect to the test data
            with tf.GradientTape(watch_accessed_variables=False) as test_tape:
                test_tape.watch(model_copy.trainable_variables)
                test_pred = model_copy(test_input)
                test_loss = tf.reduce_mean(c_loss(test_target, test_pred))
            g = test_tape.gradient(test_loss, model_copy.trainable_variables)
            task_gradients.append(g)
        # Meta-update: apply the accumulated gradients to the original model
        if task_gradients:
            sum_gradients = [tf.reduce_mean(tf.stack([grads[layer] for grads in task_gradients]), axis=0)
                             for layer in range(len(model.trainable_variables))]
            optim_test.apply_gradients(zip(sum_gradients, model.trainable_variables))
        total_l += test_loss.numpy()
        loss_evol = total_l/(step+1)
        losses.append(loss_evol)
        if step % log_step == 0:
            print(f'Meta step: {step}. Loss: {loss_evol}')
    return model, losses
```

## REFERENCES
[1] Finn, C., Abbeel, P. &amp; Levine, S.. (2017). Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks. <i>Proceedings of the 34th International Conference on Machine Learning</i>, in <i>Proceedings of Machine Learning Research</i> 70:1126-1135 Available from https://proceedings.mlr.press/v70/finn17a.html and https://proceedings.mlr.press/v70/finn17a/finn17a.pdf.
