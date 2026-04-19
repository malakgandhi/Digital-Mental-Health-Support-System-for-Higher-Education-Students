# Digital-Mental-Health-Support-System-for-Higher-Education-Students
This project is a web-based mental health support system designed to identify possible depresison risk among higher-education students. It aims to spot early awareness by analyzing student-related factors such as academic pressure, study satisfaction, sleep duration, dietary habits, financial stress and family history of mental illness.
The main goal of this project is to provide a simple and accessible tool that can help students, especially those in rural or suburban colleges, become more aware of their mental health condition and seek support when needed.

## Objective
The objective of this project is to detect possible signs of depression using machine learning and provide users with a quick risk assessment based on their responses. This system is intended for awareness and educational purposes only.

## Key Features
- Takes student mental-health-related inputs through a web form
- Predicts possible depression risk using a trained machine learning model.
- Displays the result in a simple and understandable format
- Stores submitted responses and predictions for future analysis

## How the system works
1. The user fills out the form with personal and lifestyle related details.
2. The system processes the input values.
3. The trained machine learning model predicts the result.
4. The result is displayed on a separate page as high risk or low risk.

## Input fields used
- Gender
- Age
- Academic Pressure
- Study Satisfaction
- Sleep Duration
- Dietary Habits
- Suicidal Thoughts Recieved
- Study Hours
- Financial Stress
- Family History of Mental Illness

## Model Information
This project uses machine learning to predict the depression risj. Different models were tested, and the best performing model was saved and used in the Flask application for prediction

## Limitations
- This system is not a medical diagnosis tool.
- The prediction may not be always be fully accurate, precise or sensitive.
- The result should not replace professional mental health advice.

## Disclaimer
This project is created for educational and awareness purposes only. If a person is experiency symptoms of depression, anxiety or any other mental health condition, they should visit a qualified psychologist, psychiatrist, or mental health professional.

## Future Scope
- Improve the user interface
- Add support for more mental health conditions
- Improve model accuraccy with better data
- Store data in a database rather instead of a CSV file
- Add support resources or emergency help links