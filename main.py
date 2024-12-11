import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from flask import Flask, request, render_template_string
import json

# Load the dataset
data_file_path = "software_cost_dataset.csv"
try:
    df = pd.read_csv(data_file_path)
except FileNotFoundError:
    raise Exception(f"Dataset file '{data_file_path}' not found. Ensure the file exists and is accessible.")

# Prepare features and target variable
features = [
    "Project_Size",
    "Project_Duration",
    "Team_Size",
    "Complexity",
    "Reliability",
    "Database_Size",
    "Team_Cohesion",
    "Developer_Experience",
    "Software_Tools",
]

if not all(feature in df.columns for feature in features + ["Project_Cost"]):
    raise Exception("Dataset is missing required columns. Ensure the dataset contains all required fields.")

X = df[features]
y = df["Project_Cost"]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model using RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R² Score: {r2}")

# Load feature details from an external JSON file
feature_details_file = "feature_details.json"
try:
    with open(feature_details_file, "r") as f:
        feature_details = json.load(f)
except FileNotFoundError:
    raise Exception(f"Feature details file '{feature_details_file}' not found. Ensure the file exists and is accessible.")
except json.JSONDecodeError:
    raise Exception(f"Error decoding JSON from '{feature_details_file}'. Ensure the file is properly formatted.")

app = Flask(__name__)

# Define the valid input ranges for each feature
input_ranges = {
    'Project_Size': (500, 5000),
    'Project_Duration': (3, 24),
    'Team_Size': (3, 20),
    'Complexity': (1, 10),
    'Reliability': (1, 5),
    'Database_Size': (100, 1000),
    'Team_Cohesion': (1, 5),
    'Developer_Experience': (1, 5),
    'Software_Tools': (1, 5)
}

@app.route("/")
def home():
    html_template = """
    <html>
    <head>
        <title>Project Cost Estimation Tool</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                background-color: #f4f4f4;
                color: #333;
                margin: 0;
                padding: 20px;
            }
            h1 {
                text-align: center;
            }
            form {
                max-width: 600px;
                margin: auto;
                padding: 20px;
                background: white;
                border-radius: 8px;
                box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
            }
            label {
                display: block;
                margin-bottom: 5px;
            }
            input[type="number"] {
                width: calc(100% - 10px);
                padding: 8px;
                margin-bottom: 15px;
                border-radius: 4px;
                border: 1px solid #ccc;
            }
            input[type="submit"] {
                background-color: #28a745;
                color: white;
                padding: 10px;
                border: none;
                border-radius: 4px;
                cursor: pointer;
            }
            input[type="submit"]:hover {
                background-color: #218838;
            }
            .details-link {
                font-size: 0.9em;
                color: #007bff;
                text-decoration: none;
                margin-left: 5px;
            }
            .details-link:hover {
                text-decoration: underline;
            }
            .hint {
                font-size: 0.85em;
                color: #555;
                margin-top: 5px;
                display: none; /* Initially hidden */
            }
            input[type="number"]:focus + .hint {
                display: block; /* Show hint when input is focused */
            }
        </style>
    </head>
    <body>
        <h1><a href="/details/COCOMO II" style="text-decoration: none; color: inherit;">AI Project Cost Estimation Tool Using COCOMO-II</a></h1>
        <form action="/predict" method="post">
            {% for feature in features %}
            <label>{{ feature.replace('_', ' ') }}: <a class="details-link" href="/details/{{ feature }}">Learn more</a></label>
            <input type="number" name="{{ feature }}" required>
            <p class="hint" id="{{ feature }}_hint">
                {% if feature == 'Project_Size' %}
                    Enter a value between 500 and 5000 function points.
                {% elif feature == 'Project_Duration' %}
                    Enter a value between 3 and 24 months.
                {% elif feature == 'Team_Size' %}
                    Enter a value between 3 and 20 members.
                {% elif feature == 'Complexity' %}
                    Enter a value between 1 (low complexity) and 10 (high complexity).
                {% elif feature == 'Reliability' %}
                    Enter a value between 1 (low reliability) and 5 (high reliability).
                {% elif feature == 'Database_Size' %}
                    Enter a value between 100 and 1000 GB.
                {% elif feature == 'Team_Cohesion' %}
                    Enter a value between 1 (low cohesion) and 5 (high cohesion).
                {% elif feature == 'Developer_Experience' %}
                    Enter a value between 1 (low experience) and 5 (high experience).
                {% elif feature == 'Software_Tools' %}
                    Enter a value between 1 (basic tools) and 5 (advanced tools).
                {% endif %}
            </p>
            {% endfor %}
            <input type="submit" value="Predict Cost">
        </form>
    </body>
    </html>
    """
    return render_template_string(html_template, features=features)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Capture the user inputs for each feature
        inputs = {feature: float(request.form[feature]) for feature in features}

        # Validate inputs
        for feature, value in inputs.items():
            min_val, max_val = input_ranges[feature]
            if not (min_val <= value <= max_val):
                return f"<h1>Error: {feature.replace('_', ' ')} must be between {min_val} and {max_val}.</h1><br><a href='/'>Back</a>"

        input_df = pd.DataFrame([inputs])

        # Predict the total project cost
        predicted_cost = model.predict(input_df)[0]

        # If the predicted cost is negative, set it to zero
        if predicted_cost < 0:
            predicted_cost = 0

        # Get feature importances from the model to approximate contributions
        feature_importances = model.feature_importances_

        # Normalize the feature importances to make the sum equal to 1
        total_importance = sum(feature_importances)
        normalized_importances = [importance / total_importance for importance in feature_importances]

        # Create a table for the cost breakdown
        breakdown_table = []
        total_cost = 0

        for i, (feature, value) in enumerate(inputs.items()):
            # Calculate the contribution of each feature
            feature_cost = normalized_importances[i] * predicted_cost
            breakdown_table.append((feature.replace('_', ' ').title(), value, feature_cost))
            total_cost += feature_cost

        # Add the total row
        breakdown_table.append(("Total", "", total_cost))

        # Ensure the total cost matches the predicted cost (due to potential rounding errors)
        breakdown_table[-1] = ("Total", "", predicted_cost)

        # Prepare the HTML table for display
        table_html = "<table border='1' style='border-collapse: collapse; width: 80%; margin: auto;'><thead><tr><th>Feature</th><th>Value</th><th>Cost</th></tr></thead><tbody>"
        
        for row in breakdown_table:
            table_html += f"<tr><td>{row[0]}</td><td>{row[1]}</td><td>${row[2]:,.2f}</td></tr>"
        
        table_html += "</tbody></table>"

        # Display the result with the table and predicted total cost
        result_html = f"""
        <h1>Estimated Project Cost: ${predicted_cost:,.2f}</h1>
        <h2>Cost Breakdown</h2>
        {table_html}
        <br><a href='/'>Back</a>
        """
        return result_html
    except Exception as e:
        return f"<h1>Error: {e}</h1><br><a href='/'>Back</a>"

@app.route("/details/<feature>")
def details(feature):
    detail = feature_details.get(feature, "<strong>Error:</strong> No details available for this feature.")
    return f"""
    <html>
    <head>
        <title>{feature.replace('_', ' ').title()} Details</title>
    </head>
    <body>
        <h2>{feature.replace('_', ' ').title()}</h2>
        <p>{detail}</p>
        <br><a href='/'>Back</a>
    </body>
    </html>
    """

if __name__ == "__main__":
    app.run(debug=True)
