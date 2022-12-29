import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import tensorflow as tf
from dotenv import load_dotenv
from pydantic import BaseSettings
from pydantic.class_validators import root_validator
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification

GPU_AVAILABLE = tf.test.is_gpu_available()

load_dotenv(".env")  # need to run script from root of project to read this


class EnvVars(BaseSettings):
    """Pydantic data class to hold environment variables"""

    TRAIN_DATA_PATH: Path
    EPOCHS: int = 1
    BATCH_SIZE: int = 64 if GPU_AVAILABLE else 1
    VAL_BATCH_SIZE: int = 16 if GPU_AVAILABLE else 1
    SAVE_MODEL_DIR: Optional[Path] = None

    @root_validator
    def _convert_relative_to_absolute_paths(cls, values):
        values["TRAIN_DATA_PATH"] = values["TRAIN_DATA_PATH"].resolve()
        if values["SAVE_MODEL_DIR"] is not None:
            values["SAVE_MODEL_DIR"] = values["SAVE_MODEL_DIR"].resolve()
        return values


ENV_VARS = EnvVars()


def custom_training(
    model, tokenizer, x, y, epochs=1, batch_size=64, val_batch_size=16, save_model_dir=None, stats_dir=None, val_step_freq=None
):
    """Implements a custom training loop so can validate more often"""
    start_time = time.time()
    train_texts, val_texts, train_labels, val_labels = train_test_split(x, y, test_size=0.05)

    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True)
    print("Found all tokens in {} min".format((time.time() - start_time) / 60))

    # Make tf dataset objects
    train_dataset = tf.data.Dataset.from_tensor_slices((dict(train_encodings), train_labels))
    train_dataset = train_dataset.shuffle(1000).batch(batch_size)
    val_dataset = tf.data.Dataset.from_tensor_slices((dict(val_encodings), val_labels))
    val_dataset = val_dataset.shuffle(1000).batch(val_batch_size)

    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
    loss_func = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    train_losses = []
    val_losses = []
    steps = int(np.ceil(len(train_dataset)))
    if not val_step_freq:
        val_step_freq = max(1, int(steps / 10))
    for epoch in range(epochs):
        print("Starting epoch {}/{}".format(epoch + 1, epochs))
        for step, (x_batch, y_batch) in enumerate(train_dataset):
            step_time = time.time()
            train_loss = train_step(model, x_batch, y_batch, loss_func, optimizer)
            print("Training loss: {}".format(train_loss))
            train_losses.append([epoch, step, steps, float(train_loss.numpy())])

            if step % val_step_freq == 0:
                val_time = time.time()

                # Validate every certain number of steps
                total_val_loss = validation_step(model, val_dataset, loss_func) / len(val_dataset)
                val_losses.append([epoch, step, steps, float(total_val_loss.numpy())])
                print("Validation loss: {}, time {}".format(total_val_loss, time.time() - val_time))
            print("Step {}/{} Step time: {}".format(step + 1, steps, time.time() - step_time))

    print("Training time: {}".format(time.time() - start_time))
    if save_model_dir:
        print(f"Saving model in {save_model_dir}")
        model.save_pretrained(save_model_dir)
        tokenizer.save_pretrained(save_model_dir)
    if stats_dir:
        print("Saving stats")
        json.dump(
            {"train_losses": train_losses, "val_losses": val_losses},
            open(stats_dir, "w"),
        )


@tf.function
def validation_step(model, val_dataset, loss_func):
    """Performs one pass through validation set"""
    total_loss = tf.constant(0, dtype=tf.float32)
    for i, (x_batch_val, y_batch_val) in enumerate(val_dataset):
        val_logits = model(x_batch_val, training=False)
        val_loss = loss_func(y_batch_val, val_logits.logits)
        total_loss += val_loss

    return total_loss  # tf.math.divide(total_loss, tf.constant(len(val_dataset), dtype=tf.float32))


@tf.function  # comment out to run in debug
def train_step(model, x, y, loss_func, optimizer):
    """Implements one training step. Use decorator for fast performance using graphs. Seems to halve the train time!"""
    # Open a GradientTape to records operations during the forward pass which enables auto-differentiation
    with tf.GradientTape() as tape:
        # Run forward pass, calculate loss
        train_logits = model(x, training=True)
        train_loss = loss_func(y, train_logits.logits)
    # Use the tape tp automatically retrieve the gradients of the trainable variables with respect to the loss
    grads = tape.gradient(train_loss, model.trainable_weights)
    # Run one step of gradient descent by updating the values of the variables to minimise the loss
    optimizer.apply_gradients(zip(grads, model.trainable_weights))

    return train_loss


def main() -> None:
    """Train model"""
    with open(ENV_VARS.TRAIN_DATA_PATH, "r") as file:
        train_data = json.load(file)

    model = TFDistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    x_train, y_train = [list(x) for x in zip(*[[t["text"], t["label"]] for t in train_data])]

    custom_training(
        model,
        tokenizer,
        x_train,
        y_train,
        epochs=ENV_VARS.EPOCHS,
        batch_size=ENV_VARS.BATCH_SIZE,
        val_batch_size=ENV_VARS.VAL_BATCH_SIZE,
        save_model_dir=ENV_VARS.SAVE_MODEL_DIR,
    )


if __name__ == "__main__":
    main()
