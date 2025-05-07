# scripts/prepare_dataset.py
import json
import random
from sklearn.model_selection import train_test_split
import os


def load_and_split_dataset(json_path, output_dir, test_size=0.2, seed=42):
    with open(json_path, 'r') as f:
        data = json.load(f)

    male_entries = []
    female_entries = []

    for item in data:
        male_entries.append({
            "input": f"<Question:> {item['question_male']} <Answer:> {item['biased_answer_male']}",
            "target": item['unbiased_answer_male']
        })
        female_entries.append({
            "input": f"<Question:> {item['question_female']} <Answer:> {item['biased_answer_female']}",
            "target": item['unbiased_answer_female']
        })

    # Split male and female separately
    male_train, male_test = train_test_split(male_entries, test_size=test_size, random_state=seed)
    female_train, female_test = train_test_split(female_entries, test_size=test_size, random_state=seed)

    train_data = male_train + female_train
    test_data = male_test + female_test

    random.shuffle(train_data)
    random.shuffle(test_data)

    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, 'train.json'), 'w') as f:
        json.dump(train_data, f, indent=2)

    with open(os.path.join(output_dir, 'test.json'), 'w') as f:
        json.dump(test_data, f, indent=2)

    print(f"✅ Dataset split completed. Saved to {output_dir}")


if __name__ == "__main__":
    load_and_split_dataset("../data/data.json", "../data/")