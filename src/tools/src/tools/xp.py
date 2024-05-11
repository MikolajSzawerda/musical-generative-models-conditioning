import os
import pandas as pd
import datetime
import uuid
class AudiocraftInference:
    def __init__(self, project_base, experiment_name, model, file_saver):
        self.experiment_name = experiment_name
        self.model = model
        self.save_file = file_saver
        self.base_path = os.path.join(project_base, 'data', 'experiments', experiment_name)
        self.data_path = os.path.join(self.base_path, 'data')
        self.results = []  # In-memory storage for results
        
        # Ensure directories exist
        os.makedirs(self.data_path, exist_ok=True)
        
        # Load existing results if the CSV file already exists
        self.results_file = os.path.join(self.base_path, 'results.csv')
        if os.path.exists(self.results_file):
            self.results_df = pd.read_csv(self.results_file)
        else:
            self.results_df = pd.DataFrame(columns=['prompt', 'file_path'])

    def ask(self, prompts):
        wavs = self.model.generate(prompts)
        results = []
        for i, wav in enumerate(wavs):
            file_name = uuid.uuid1()
            path = f'{self.data_path}/{file_name}'
            self.save_file(path, wav.cpu())
            results.append({
                'prompt': prompts[i],
                'file_path': f'{file_name}.wav'
            })
        self.results_df = pd.concat([self.results_df, pd.DataFrame(results)], ignore_index=True)
            
            

    def save_results_to_csv(self):
        """ Save the DataFrame to a CSV file """
        self.results_df.to_csv(self.results_file, index=False)
    
    