from datasets import load_dataset, concatenate_datasets, Dataset
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", type=str,
                        help="Path to store data")
    args = parser.parse_args() 
    
    # 사용 가능한 모든 config
    configs = [
        'claude_evol', 'code-alpaca', 'helpsteer', 'ko_wikidata',
        'mmlu_abstract_algebra', 'mmlu_all', 'mmlu_anatomy', 'mmlu_astronomy',
        'mmlu_business_ethics', 'mmlu_clinical_knowledge', 'mmlu_college_chemistry', 
        'mmlu_econometrics', 'mmlu_formal_logic', 'mmlu_global_facts',
        'mmlu_high_school_european_history', 'mmlu_high_school_geography', 'mmlu_high_school_microeconomics', 
        'mmlu_high_school_government_and_politics', 'mmlu_high_school_macroeconomics',
        'mmlu_high_school_psychology', 'mmlu_high_school_us_history',
        'mmlu_high_school_world_history', 'mmlu_human_aging', 'mmlu_human_sexuality',
        'mmlu_international_law', 'normal_instructions', 'tiny-textbooks'
        # 'mmlu_high_school_mathematics', 'mmlu_high_school_physics', 'mmlu_high_school_statistics', 
        # 'mmlu_college_computer_science', 'mmlu_college_mathematics',
        # 'mmlu_college_medicine', 'mmlu_college_physics', 'mmlu_computer_security',
        # 'mmlu_conceptual_physics', 'mmlu_electrical_engineering', 'mmlu_college_biology',
        # 'mmlu_elementary_mathematics',
        # 'mmlu_high_school_biology', 'mmlu_high_school_chemistry', 'mmlu_high_school_computer_science',
    ]

    # train, validation, test 모두 병합
    all_train_data = []
    all_validation_data = []
    all_test_data = []

    for config in configs:
        dataset = load_dataset('maywell/korean_textbooks', config)
        if 'train' in dataset:
            all_train_data.append(dataset['train'])
        # All configs have no validation and no tests. 
        # if 'validation' in dataset:
        #     all_validation_data.append(dataset['validation'])
        # if 'test' in dataset:
        #     all_test_data.append(dataset['test'])

    # 각각 병합
    train_dataset = concatenate_datasets(all_train_data) if all_train_data else None
    # validation_dataset = concatenate_datasets(all_validation_data) if all_validation_data else None
    # test_dataset = concatenate_datasets(all_test_data) if all_test_data else None

    korean_textbook = {'text':train_dataset['text']}
    korean_textbook = Dataset.from_dict(korean_textbook)

    korean_textbook.save_to_disk(args.save_path)
    print(f"Korean textbooks savd in {args.save_path}")

if __name__ == "__main__":
    main()