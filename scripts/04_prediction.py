import pandas as pd
import joblib

def predict(input_features, model_path):
    model = joblib.load(model_path)
    df = pd.DataFrame([input_features])
    prediction = model.predict(df)
    return prediction[0]

if __name__ == "__main__":
    sample_input = {
        "GRE_Score": 0.9, "TOEFL_Score": 0.85, "University_Rating": 4,
        "SOP": 4.5, "LOR": 4.5, "CGPA": 0.95, "Research": 1
    }
    pred = predict(sample_input, "../outputs/admission_model.pkl")
    print(f"Predicted Admission Chance: {pred:.3f}")
