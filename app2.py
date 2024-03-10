import gradio as gr
import pandas as pd
import pickle

from train_model.train import runner
# Define a custom CSS style for the Gradio interface

css = """
body {
    background-color: #222;
    color: #eee;
}

.gradio-interface {
    background-color: #333;
    border-radius: 10px;
    padding: 20px;
}

.gradio-interface .output-container {
    background-color: #444;
    padding: 10px;
    margin-bottom: 10px;
}

.gradio-interface .output-container img {
    max-width: 100%;
}
"""

with open("C:/Users/hp/Credit-Default-Risk-Prediction/models/cdr_model.pickle", "rb") as f:
    random_model = pickle.load(f)

# Function to make predictions using the classifier model
models = [random_model]
def predict_credit_risk(person_age,person_income,person_home_ownership,person_emp_length,loan_intent,loan_grade,loan_amnt,loan_int_rate,loan_percent_income,cb_person_default_on_file, cb_person_cred_hist_length):
    # Preprocess input features if needed
    
    feature_names = ["person_age","person_income","person_home_ownership","person_emp_length","loan_intent","loan_grade","loan_amnt","loan_int_rate","loan_percent_income","cb_person_default_on_file","cb_person_cred_hist_length"]
    input_data = {
    'person_age': person_age,
    'person_income': person_income,
    'person_home_ownership': person_home_ownership,
    'person_emp_length': person_emp_length,
    'loan_intent': loan_intent,
    'loan_grade': loan_grade,
    'loan_amnt': loan_amnt,
    'loan_int_rate': loan_int_rate,
    'loan_percent_income': loan_percent_income,
    'cb_person_default_on_file': cb_person_default_on_file,
    'cb_person_cred_hist_length': cb_person_cred_hist_length
}
    # print(input_data)
    
    # Convert input data to a DataFrame with a single row
    input_data = pd.DataFrame([input_data])
    input_data = input_data[feature_names]
    
        
    input_data = input_data.fillna(input_data.median())
    input_data=input_data.drop_duplicates()
    
    # print(input_data)
    
    prediction = models[0].predict(input_data)
    
    if prediction[0] == 0:
        return "No Credit Default Risk Detected✅"
    else:
        return "Credit Default Risk Detected ⚠️"
  
# Create Gradio interface
def app_interface():# Custom CSS
    with gr.Blocks(css = css) as interface:
        gr.HTML("<img src='https://i.ibb.co/Bw08434/logo-1.png' alt='Logo' style='width:230px;height:100px;border-radius:5px;box-shadow:2px 2px 5px 0px rgba(0,0,0,0.75);background-color:black;'><br>",)
    
        with gr.Row("Credit Default Risk Prediction"):
            with gr.Column("Model Training "):
                gr.HTML("<h2>Train your own model!</h2>")
        
                gradient_input = [
                    gr.Slider(minimum=10, maximum=500, step = 5, label="Number of Estimators"),
                    gr.Slider(minimum=0.00000000001, maximum=1, label="Learning Rate", step = 0.01),
                    gr.Slider(minimum=0.00000000001, maximum=1, label="Gamma", step = 0.2),
                    gr.Slider(minimum=5, maximum=25, label="Max Depth", step = 1),
                    gr.Slider(minimum=0.00000000001, maximum=1, label="Test Size", step= 0.1)]
                    
                
                gradient_output = [
                    gr.Textbox(label="Accuracy Score"),
                    gr.Textbox(label="Precision Score"),
                    gr.Textbox(label="Recall Score"),
                    gr.Textbox(label="F1 Score"),
                    gr.Image(label="ROC Curve"),
                    gr.Image(label="Learning Curve")
                    
                ]
                    
                gradient_train_button = gr.Button(value="Train Gradient Boost Regressor Model")

                gr.HTML("<h3>Dataset link here: <a href='https://www.kaggle.com/datasets/laotse/credit-risk-dataset'>Dataset</a>.</h3>")
                    
            with gr.Column("Please fill the form to predict credit default risk!"):
                gr.HTML("<h2>Please fill the form to predict credit default risk!</h2>")
                
                inp = [
                    gr.Slider(label="Age", minimum=1, maximum=120),
                    gr.Slider(label="Income", minimum=100, maximum=6000000, step = 1000),
                    gr.Radio(label="Home Ownership", choices = [("Rent", 0), ("Own", 1), ("Mortgage", 2), ("Other", 3)]),
                                
                    gr.Slider(label="Employment Length", minimum=0, maximum=135, step=3),
                    gr.Radio(label="Loan Intent", choices=[
                        ('Personal', 0),( 'Education',1), ('Medical',2), ('Venture',3), ('Home Improvement',4),
                        ('Debt Consolidation',5)]),
                    gr.Radio(label="Loan Grade", choices=[
                        ('A', 0),( 'B',1), ('C',2), ('D',3), ('E',4),('F',5),('G', 6)]),
                    
                    gr.Slider(label="Loan Amount", minimum=0, maximum=500000, step=1000),
                    gr.Slider(label="Loan Interest Rate", minimum=0, maximum=100, step=5),
                    gr.Slider(label="Loan Percent Income", minimum=0, maximum=100, step=1),
                    
                    gr.Radio(label="Default History", choices=[("Yes",1), ("No", 0)]),
                    gr.Slider(label="Credit History Length", minimum=0, maximum=100, step=5),
        
                ]
                
                output = [gr.Textbox(label="Prediction")]
                predict_button = gr.Button(value="Credit Default Risk Prediction")
                
        gradient_train_button.click(runner, inputs=gradient_input, outputs=gradient_output)
        predict_button.click(predict_credit_risk, inputs=inp, outputs=output)

    interface.launch()

if __name__ == "__main__":
    app_interface()