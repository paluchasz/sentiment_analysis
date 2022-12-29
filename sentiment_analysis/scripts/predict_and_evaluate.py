import json
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from dotenv import load_dotenv
from pydantic.class_validators import root_validator
from pydantic.env_settings import BaseSettings
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

GPU_AVAILABLE = tf.test.is_gpu_available()

load_dotenv(".env")  # need to run script from root of project to read this


class EnvVars(BaseSettings):
    """Pydantic data class to hold environment variables"""

    TEST_DATA_PATH: Path
    TRAINED_MODEL_DIR: Path
    BATCH_SIZE: int = 64 if GPU_AVAILABLE else 1

    @root_validator
    def _convert_relative_to_absolute_paths(cls, values):
        values["TEST_DATA_PATH"] = values["TEST_DATA_PATH"].resolve()
        values["TRAINED_MODEL_DIR"] = values["TRAINED_MODEL_DIR"].resolve()
        return values


ENV_VARS = EnvVars()


def predict_and_evaluate(model, tokenizer, x, y, batch_size=64):
    """Predict and evaluate on the test set"""
    predictions = []
    batches_num = int(np.math.ceil(len(x) / batch_size))
    print("Number of batches {} of size {}".format(batches_num, batch_size))
    start_time = time.time()
    for i in range(batches_num):
        print("Batch {},".format(i), end=" ")
        batch_start_time = time.time()
        input_tokens = tokenizer(
            x[i * batch_size : min((i + 1) * batch_size, len(x))],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="tf",
        )
        print("Tokenizer time: {},".format(time.time() - batch_start_time), end=" ")
        tf_outputs = model(input_tokens)
        predictions += tf.nn.softmax(tf_outputs.logits, axis=-1).numpy().tolist()
        print("Batch time: {}".format(time.time() - batch_start_time))

    print("Time taken: {} minutes ".format((time.time() - start_time) / 60))
    print("Predictions: ", predictions)
    predictions = [1 if p[1] > 0.5 else 0 for p in predictions]
    print("Predictions: ", predictions)
    print("Actual:      ", y)
    accuracy = sum([1 for i in range(len(predictions)) if predictions[i] == y[i]]) / len(predictions)
    print("Accuracy: {}".format(accuracy))
    return predictions


def main() -> None:
    """Evaluate the current model on the test set"""
    with open(ENV_VARS.TEST_DATA_PATH, "r") as file:
        test_data = json.load(file)
    x_test, y_test = [list(x) for x in zip(*[[t["text"], t["label"]] for t in test_data])]

    tokenizer = AutoTokenizer.from_pretrained(ENV_VARS.TRAINED_MODEL_DIR)
    model = TFAutoModelForSequenceClassification.from_pretrained(ENV_VARS.TRAINED_MODEL_DIR)

    predict_and_evaluate(model, tokenizer, x_test, y_test)


if __name__ == "__main__":
    main()
