import argparse
import json
import pandas as pd
import random
from datasets import Dataset
from huggingface_hub import HfApi
from itertools import chain


def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    print("Structure of loaded data:")
    if isinstance(data, list):
        print(f"List with {len(data)} items")
        if data:
            print("First item keys:")
            print(json.dumps(list(data[0].keys()), indent=2))
    elif isinstance(data, dict):
        print("Dictionary keys:")
        print(json.dumps(list(data.keys()), indent=2))
    else:
        print(f"Unexpected data type: {type(data)}")
    return data

def save_jsonl(data, file_path):
    with open(file_path, 'w') as outfile:
        for item in data:
            json.dump(item, outfile)
            outfile.write('\n')

def extract_model_info(json_data):
    extracted_info = []
    for entry in json_data:
        if 'datasets' in entry:
            for dataset in entry['datasets']:
                dataset_name = dataset.get('dataset', '')
                if 'sota' in dataset and 'metrics' in dataset['sota'] and 'rows' in dataset['sota']:
                    table_metric = dataset['sota']['metrics']
                    for row in dataset['sota']['rows']:
                        model_info = {
                            'dataset': dataset_name,
                            'model_name': row.get('model_name', ''),
                            'full_name': row.get('model_name', ''),  # Using model_name as full_name
                            'paper_title': row.get('paper_title', ''),
                            'metrics': row.get('metrics', {}),
                            'table_metrics': table_metric,
                            'area': entry.get('categories', []),  # Using categories as area
                        }
                        extracted_info.append(model_info)
        elif 'full_name' in entry:
            # This is for the methods.json file
            model_info = {
                'full_name': entry.get('full_name', ''),
                'description': entry.get('description', ''),
                'area': [area['area'] for area in entry.get('collections', []) if 'area' in area],
            }
            extracted_info.append(model_info)
    return extracted_info


def create_mixed_model_dataframe(df):
    mixed_models = []
    unique_areas = set(chain.from_iterable(filter(None, df['area'])))
    for area in unique_areas:
        same_area_models = df[df['area'].apply(lambda x: area in x)]['full_name'].tolist()
        different_area_models = df[df['area'].apply(lambda x: area not in x)]['full_name'].tolist()
        for model in same_area_models:
            other_same_area_models = [m for m in same_area_models if m != model]
            if len(other_same_area_models) >= 2:
                selected_same_area_models = random.sample(other_same_area_models, 2)
            else:
                continue
            if not different_area_models:
                continue
            selected_model_from_diff_area = random.choice(different_area_models)
            model_combo = [model] + selected_same_area_models + [selected_model_from_diff_area]
            random.shuffle(model_combo)
            
            # Modified area assignment
            areas = []
            for m in model_combo:
                model_areas = df.loc[df['full_name'] == m, 'area']
                if not model_areas.empty:
                    model_area = model_areas.values[0]
                    areas.append(area if area in model_area else 'Outlier')
                else:
                    areas.append('Unknown')  # Handle case where model is not found
            
            mixed_models.append([model_combo, selected_model_from_diff_area, areas])
    
    mixed_model_df = pd.DataFrame(mixed_models, columns=['Model Combo', 'Outlier Model', 'Areas'])
    return mixed_model_df

def create_metrics_benchmark(data):
    df = pd.DataFrame(data)
    df['prompts'] = df.apply(lambda row: f"What metrics were used to measure the {row.get('model_name', 'unknown model')} model in the {row.get('paper_title', 'unnamed paper')} paper on the {row.get('dataset', 'unknown dataset')} dataset?", axis=1)
    df['metrics_response'] = df['table_metrics'].apply(lambda x: ', '.join(x) if isinstance(x, list) else str(x))
    return df[['prompts', 'metrics_response']]

def create_abstract_benchmark(data):
    df = pd.DataFrame(data)
    df['prompts'] = df['title'].apply(lambda x: f"Given the following ArXiv paper title {x}, write the abstract for the paper")
    return df[['prompts', 'abstract']]

def create_model_description_benchmark(data):
    df = pd.DataFrame(data)
    df['prompts'] = df['full_name'].apply(lambda x: f"Given the following machine learning model name: {x}, provide a description of the model")
    return df[['prompts', 'description']]

def create_research_area_benchmark(data):
    # Extract area information
    area = []
    for i in data:
        out = [x['area'] for x in i.get('collections', []) if 'area' in x]
        area.append(out)

    df = pd.DataFrame(data)
    df['area'] = area
    print("Columns in research area DataFrame:", df.columns)
    print("First few rows of research area DataFrame:")
    print(df.head())
    # Filter out rows where 'area' contains an empty string or 'General'
    df = df[~df['area'].apply(lambda x: 'General' in x or not x)]
    df['area'] = df['area'].apply(lambda x: list(set(x)))
    df['prompts'] = df['full_name'].apply(lambda x: f"Given the following machine learning model name: {x}, predict one or more research areas from the following list: [Computer Vision, Sequential, Reinforcement Learning, Natural Language Processing, Audio, Graphs]")
    
    return df[['prompts', 'area']]


def create_odd_man_out_benchmark(data):
    # Extract area information
    area = []
    for i in data:
        out = [x['area'] for x in i.get('collections', []) if 'area' in x]
        area.append(out)

    df = pd.DataFrame(data)
    df['area'] = area

    print("Columns in odd man out DataFrame:", df.columns)
    print("First few rows of odd man out DataFrame:")
    print(df.head())

    # Filter out rows where 'area' contains an empty string or 'General'
    df = df[~df['area'].apply(lambda x: 'General' in x or not x)]
    df = df[~df['area'].apply(lambda x: 'Sequential' in x or not x)]
    
    df['area'] = df['area'].apply(lambda x: list(set(x)))
    
    print("Unique areas:", set(chain.from_iterable(df['area'])))
    print("Number of models after filtering:", len(df))

    if len(df) < 4:
        print("Warning: Not enough models to create mixed model dataframe")
        return pd.DataFrame(columns=['prompts', 'Outlier Model'])

    # Create mixed model dataframe
    try:
        mixed_model_df = create_mixed_model_dataframe(df)
    except Exception as e:
        print(f"Error in create_mixed_model_dataframe: {e}")
        return pd.DataFrame(columns=['prompts', 'Outlier Model'])
    
    mixed_model_df['prompts'] = mixed_model_df['Model Combo'].apply(lambda x: f"Given the following list of machine learning models: {x}, select the one that least belongs in the list.")
    return mixed_model_df[['prompts', 'Outlier Model']]

def create_similar_models_benchmark(data):
    df = pd.DataFrame(data)
    grouped = df.groupby('dataset')['model_name'].apply(list).reset_index()
    filtered = grouped[grouped['model_name'].apply(len) > 1]
    filtered = filtered.dropna(subset=['dataset'])
    filtered['query_model'] = filtered['model_name'].apply(lambda x: x.pop(0) if x else None)
    filtered['prompts'] = filtered.apply(lambda row: f"Given the following machine learning model name: {row['query_model']}, and dataset: {row['dataset']}, provide a list of other models that have been benchmarked on that dataset", axis=1)
    return filtered[['prompts', 'model_name']]

def create_top_models_benchmark(data):
    df = pd.DataFrame(data)
    grouped = df.groupby('dataset')['model_name'].apply(list).reset_index()
    filtered = grouped[grouped['model_name'].apply(len) > 1]
    filtered = filtered.dropna(subset=['dataset'])
    filtered['prompts'] = filtered['dataset'].apply(lambda x: f"Given the following benchmark dataset: {x}, provide a list of best performing models on that benchmark. Provide specific model names.")
    return filtered[['prompts', 'model_name']]

def create_dataset_description_benchmark(data):
    df = pd.DataFrame(data)
    
    print("Columns in dataset description DataFrame:", df.columns)
    print("First few rows of dataset description DataFrame:")
    print(df.head())

    # Filter out rows where description contains "Click to add a brief description"
    df = df[~df['description'].str.contains("Click to add a brief description", na=False)]

    # Clean up the description
    df['description'] = df['description'].str.split("\r\n\r\nSource").str[0]

    # Create prompts using 'name' instead of 'dataset_name'
    df['prompts'] = df['name'].apply(lambda x: f"Given the following benchmark dataset: {x}, provide a description of the benchmark dataset")
    
    return df[['prompts', 'description']]

def create_modality_benchmark(data):
    df = pd.DataFrame(data)
    
    print("Columns in modality DataFrame:", df.columns)
    print("First few rows of modality DataFrame:")
    print(df.head())

    # Filter rows where modalities is not empty
    df = df[df['modalities'].notna() & (df['modalities'].str.len() > 0)]

    # Keep only rows where modalities contains desired items
    desired_items = {'Graphs', 'Images', 'Texts', 'Tabular', 'Videos', 'Audio'}
    df = df[df['modalities'].apply(lambda x: any(item in desired_items for item in x))]

    df['prompts'] = df['name'].apply(lambda x: f"Given the following benchmark dataset: {x}, predict one or more research areas from the following list: ['Images', 'Graphs', 'Texts', 'Tabular', 'Videos', 'Audio']")
    
    return df[['prompts', 'modalities']]

def save_benchmark(df, output_file):
    data = []
    for _, row in df.iterrows():
        output = {
            'input': [
                {"role": "system", "content": "You are a knowledgeable and helpful AI researcher"},
                {"role": "user", "content": row.get('prompts', 'No prompt available')}
            ],
            'ideal': row.get('metrics_response', 'No response available')
        }
        data.append(output)
    save_jsonl(data, output_file)

def push_to_hub(df, dataset_name):
    dataset = Dataset.from_pandas(df)
    dataset = dataset.filter(lambda x: all(v is not None for v in x.values()))
    dataset.push_to_hub(f"ArtifactAI/{dataset_name}")

def main():
    parser = argparse.ArgumentParser(description="Create ML benchmarks")
    parser.add_argument("--input_dir", required=True, help="Directory containing input JSON files")
    parser.add_argument("--output_dir", required=True, help="Directory to save output JSONL files")
    parser.add_argument("--push_to_hub", action="store_true", help="Push datasets to Hugging Face Hub")
    args = parser.parse_args()

    benchmarks = {
        "metrics": (f"{args.input_dir}/evaluation-tables.json", create_metrics_benchmark, extract_model_info),
        "abstract": (f"{args.input_dir}/papers-with-abstracts.json", create_abstract_benchmark, None),
        "model_description": (f"{args.input_dir}/methods.json", create_model_description_benchmark, None),
        "research_area": (f"{args.input_dir}/methods.json", create_research_area_benchmark, None),
        "odd_man_out": (f"{args.input_dir}/methods.json", create_odd_man_out_benchmark, None),
        "similar_models": (f"{args.input_dir}/evaluation-tables.json", create_similar_models_benchmark, extract_model_info),
        "top_models": (f"{args.input_dir}/evaluation-tables.json", create_top_models_benchmark, extract_model_info),
        "dataset_description": (f"{args.input_dir}/datasets.json", create_dataset_description_benchmark, None),
        "modality": (f"{args.input_dir}/datasets.json", create_modality_benchmark, None)
    }

    for name, (input_file, create_func, preprocess_func) in benchmarks.items():
        print(f"Creating {name} benchmark...")
        try:
            data = load_json(input_file)
            if preprocess_func:
                data = preprocess_func(data)
            
            print(f"Sample data item:")
            print(json.dumps(data[0] if isinstance(data, list) else data, indent=2))
            
            df = create_func(data)
            
            print(f"DataFrame columns: {df.columns}")
            print(f"DataFrame shape: {df.shape}")
            print("First few rows of the DataFrame:")
            print(df.head())

            output_file = f"{args.output_dir}/{name}_benchmark.jsonl"
            save_benchmark(df, output_file)
            print(f"Saved {name} benchmark to {output_file}")

            if args.push_to_hub:
                push_to_hub(df, f"{name}_benchmark")
                print(f"Pushed {name} benchmark to Hugging Face Hub")
        except Exception as e:
            print(f"Error processing {name} benchmark: {e}")
            import traceback
            traceback.print_exc()
            continue

if __name__ == "__main__":
    main()