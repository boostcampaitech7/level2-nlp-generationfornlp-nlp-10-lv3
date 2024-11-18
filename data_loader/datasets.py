from datasets import Dataset



class BaseDataset:
    def __init__(self, train_data, val_data,
                 tokenizer, template, max_length=1024):
        self.tokenizer = tokenizer
        self.template = template
        self.max_length = max_length
        self.train_data = train_data 
        self.val_data = val_data
    
    def preprocess(self):
        pass 

    def __getitem


    def len 

    # 허깅페이스 데이터셋 형태로 리턴 
    def format_data(dataset) :
        records = []
        for _, row in dataset.iterrows():
            problems = literal_eval(row['problems'])
            record = {
                'id': row['id'],
                'paragraph': row['paragraph'],
                'question': problems['question'],
                'choices': problems['choices'],
                'answer': problems.get('answer', None),
                "question_plus": problems.get('question_plus', None),
            }
            # Include 'question_plus' if it exists
            if 'question_plus' in problems:
                record['question_plus'] = problems['question_plus']
            records.append(record)
        
        df = pd.DataFrame(records)
        df['question_plus'] = df['question_plus'].fillna('')
        df['full_question'] = df.apply(lambda x: x['question'] + ' ' + x['question_plus'] if x['question_plus'] else x['question'], axis=1)
        
        dataset = Dataset.from_pandas(df)
        
        return dataset
