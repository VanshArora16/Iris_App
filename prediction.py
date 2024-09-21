import joblib
def predict(data):
    cif = joblib.load("rf_model.sav")
    return cif.predict(data)