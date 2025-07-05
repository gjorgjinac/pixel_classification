from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from autogluon_sklearn_interface import AutogluonClassifier

# Define models
models = {
    'Dummy Classifier': DummyClassifier(),
    'Decision Tree': DecisionTreeClassifier(),
    'MLP Classifier': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=100, verbose=True, early_stopping=True,
                                    solver='sgd', n_iter_no_change=5, tol=0.001),
    'Random Forest': RandomForestClassifier(n_jobs=-1, verbose=10),
    'AutoGluon': AutogluonClassifier(time_limit=3600)
}
# Fields to extract from the HDF5 file
fields_of_interest = [
    'classes', 'continent', 'dates', 'granule_id', 'latitude', 'longitude',
    'product_id', 'spectra', 'sun_azimuth_angle', 'sun_zenith_angle',
    'viewing_zenith_angle'
]
file_path = '20170710_s2_manual_classification_data.h5'
