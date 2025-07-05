from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
import argparse
from autogluon_sklearn_interface import AutogluonClassifier
from globals import *
from utils.utils_model_training import *
from utils.utils_read_data import read_data_into_dataframe

binarize_labels = False

# Ensure output directory exists
results_dir = f'results/project_split_all_binarized_{binarize_labels}'
os.makedirs(results_dir, exist_ok=True)

# Setup MLflow tracking
mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
mlflow.set_experiment(f"Pixel classification - Binarized: {binarize_labels} 5")

# read fold from command line argument, use argparse
parser = argparse.ArgumentParser(description='Train models with project split.')

parser.add_argument('--fold', type=int, default=0, help='Fold number to process.')
fold = parser.parse_args().fold
# Define models
models = {
    'Dummy Classifier': DummyClassifier(),
    'Decision Tree': DecisionTreeClassifier(),
    'MLP Classifier': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=100, verbose=True, early_stopping=True,
                                    solver='sgd', n_iter_no_change=5, tol=0.001),
    'Random Forest': RandomForestClassifier(n_jobs=-1, verbose=10),
    'AutoGluon': AutogluonClassifier(time_limit=3600)
}

# Read data and encode categorical values
df = read_data_into_dataframe(file_path, fields_of_interest)

# fix random seeds
fix_random_seed(10)
results = []

train_eval_input = prepare_fold_data(df=df, binarize_labels=binarize_labels, fold=fold)

for model_name, model in models.items():
    print(model_name)
    train_eval_output = train_and_evaluate_model(
        train_eval_input, model, model_name)
    other_info = {
        'Model': model_name,
        'Product ids': train_eval_input.test_product_ids,
        'Fold': fold
    }
    results.append({**train_eval_output.fold_results, **other_info})
    run_name = f"{model_name}_fold_{fold}"
    train_eval_output.test_and_predictions.to_csv(f'{results_dir}/{run_name}_test_and_predictions.csv')
    with mlflow.start_run(run_name=run_name):
        log_run_to_mlflow(train_eval_output, train_eval_input, fold, run_name, other_info)

    # Display results
    results_df = pd.DataFrame(results)
    results_df.to_csv(f'{results_dir}/results_fold_{fold}.csv')
