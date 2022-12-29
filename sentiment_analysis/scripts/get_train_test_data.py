from __future__ import annotations

"""
To use this script download raw data from https://ai.stanford.edu/~amaas/data/sentiment/ and specify path to the
aclImdb folder. If installed with poetry can simply run `get_train_test_data`.
"""
import json
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from pydantic import BaseSettings, root_validator

load_dotenv(".env")


class EnvVars(BaseSettings):
    """Pydantic data class to hold environment variables"""

    RAW_DATA_DIR: Path
    TRAIN_TEST_DIR: Path

    @root_validator
    def _convert_relative_to_absolute_paths(cls, values):
        values["RAW_DATA_DIR"] = values["RAW_DATA_DIR"].resolve()
        values["TRAIN_TEST_DIR"] = values["TRAIN_TEST_DIR"].resolve()
        return values


ENV_VARS = EnvVars()


def convert_raw_data(dir_path: Path) -> list[dict[str, str | int]]:
    """Convert original raw data format into train and test data files"""
    print("Converting raw data")
    examples = []
    for label_dir in ["pos", "neg"]:
        for example in Path(dir_path / label_dir).glob("*txt"):
            ex_id, ex_rating = example.stem.split("_")
            with open(example, "r") as file:
                text = file.read()
            examples.append(
                {
                    "text": text,
                    "id": int(ex_id),
                    "ex_rating": int(ex_rating),
                    "label": 1 if label_dir == "pos" else 0,
                }
            )
    np.random.shuffle(examples)
    return examples


def main() -> None:
    """Load road data and save as single train and test files"""
    train_data = convert_raw_data(Path(ENV_VARS.RAW_DATA_DIR / "aclImdb/train"))
    test_data = convert_raw_data(Path(ENV_VARS.RAW_DATA_DIR / "aclImdb/test"))

    print(f"Saving train and test data in {ENV_VARS.TRAIN_TEST_DIR}")
    with open(ENV_VARS.TRAIN_TEST_DIR / "train.json", "w") as file:
        json.dump(train_data, file)
    with open(ENV_VARS.TRAIN_TEST_DIR / "test.json", "w") as file:
        json.dump(test_data, file)


if __name__ == "__main__":
    """Doc string for check to pass. Not needed?"""
    main()
