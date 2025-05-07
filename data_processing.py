import json
import random
from sklearn.model_selection import train_test_split
from typing import List, Dict, Any
from config import config


def process_data_entry(entry: Dict[str, Any]) -> List[Dict[str, str]]:
    """Convert a single 6-field entry into two 3-field entries (male and female versions)"""
    male_version = {
        "question": entry["question_male"],
        "biased_answer": entry["biased_answer_male"],
        "unbiased_answer": entry["unbiased_answer_male"],
        "gender": "male"
    }

    female_version = {
        "question": entry["question_female"],
        "biased_answer": entry["biased_answer_female"],
        "unbiased_answer": entry["unbiased_answer_female"],
        "gender": "female"
    }

    return [male_version, female_version]


def load_and_process_data() -> Dict[str, List[Dict[str, str]]]:
    """Load raw data, process it, and split into train/test sets"""
    with open(config.raw_data_path, 'r') as f:
        raw_data = json.load(f)

    # Process all entries and flatten the list
    processed_data = []
    for entry in raw_data:
        processed_data.extend(process_data_entry(entry))

    # Split into train and test sets while maintaining gender balance
    male_data = [d for d in processed_data if d["gender"] == "male"]
    female_data = [d for d in processed_data if d["gender"] == "female"]

    male_train, male_test = train_test_split(
        male_data, test_size=config.test_size, random_state=config.random_seed
    )
    female_train, female_test = train_test_split(
        female_data, test_size=config.test_size, random_state=config.random_seed
    )

    train_data = male_train + female_train
    test_data = male_test + female_test

    # Shuffle the datasets
    random.shuffle(train_data)
    random.shuffle(test_data)

    # Save processed data
    processed_data_dict = {
        "train": train_data,
        "test": test_data
    }

    with open(config.processed_data_path, 'w') as f:
        json.dump(processed_data_dict, f, indent=2)

    return processed_data_dict


def create_prompt(question: str, biased_answer: str) -> str:
    """Create the input prompt for the model"""
    return f"""<s>[INST] Below is a question and a potentially biased answer. Please rewrite the answer to be more unbiased while still addressing the question.

<Question:> {question}
<Biased Answer:> {biased_answer}

<Unbiased Answer:> [/INST]"""


if __name__ == "__main__":
    data = load_and_process_data()
    print(f"Train samples: {len(data['train'])}")
    print(f"Test samples: {len(data['test'])}")