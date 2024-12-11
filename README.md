# Software-Project-Cost-Estimation
Using COCOMO-2 model find software project cost estimation model using python. 
1. Problem Definition
In software development, project cost estimation is crucial for proper planning, budgeting, and resource allocation. The cost of a software project is influenced by various factors such as project size, complexity, team size, and project duration, among others. Accurate cost estimation helps ensure that a project is completed within budget and on time.
The goal of this project was to develop an AI-based tool that estimates the cost of a software project based on various input features using the COCOMO-II model. This tool is intended for use by project managers to make informed decisions about the financial resources required for a project.
The input features include:
    • Project Size: The size of the software in terms of function points or lines of code.
    • Project Duration: The total time required to complete the project.
    • Team Size: The number of people involved in the project.
    • Complexity: The complexity level of the project.
    • Reliability: The desired level of reliability for the project.
    • Database Size: The size of the database that the software will handle.
    • Team Cohesion: How well the team works together.
    • Developer Experience: The experience level of the development team.
    • Software Tools: The tools and technologies used in the development.
The output is the estimated cost of completing the software project, which is critical for making appropriate decisions regarding project management and funding.
2. Method of Solving the Problem
To address this problem, we followed these steps:
    1. Data Collection: We collected historical software project data, including input features and corresponding costs, which were used to train and test machine learning models.
    2. Feature Engineering: The dataset was pre-processed to handle missing data, outliers, and standardized for the model. We ensured all necessary features were included and correctly formatted.
    3. Initial Model Selection: We started with a Linear Regression model, chosen for its simplicity and interpretability. This model relates input features to project costs and provides coefficients that help break down individual cost contributions.
    4. Issues with Linear Regression:
        ◦ Negative Cost Predictions: Linear regression produced unrealistic negative project costs.
        ◦ Cost Mismatch: The predicted costs did not match the sum of individual feature contributions.
        ◦ Linear Assumption: Linear regression struggled to model the complex, non-linear relationships in the data.
    5. Switch to Random Forest Regressor: To address these issues, we switched to the Random Forest Regressor, which better captures non-linear relationships and provides more accurate results. It uses an ensemble of decision trees, improving prediction accuracy.
    6. Cost Breakdown: To ensure the total cost matched the sum of feature contributions, we normalized the feature importances, ensuring consistency between the predicted cost and the breakdown.
    7. Web Interface: We developed a Flask web application to enable users to input feature values and receive cost estimates and breakdowns. The app allows users to input values for factors like project size, team size, and complexity, providing an easy-to-use tool for cost estimation.
3. Results Obtained
The model was evaluated using both train and test data, and the results were as follows:
    • Mean Squared Error (MSE): This metric measures the average squared difference between the predicted and actual values. Lower MSE indicates better model performance.
    • R² Score: The R² score measures how well the model's predictions match the actual data. An R² score close to 1 indicates a strong fit.
Sample Results:
    • Model Performance:
        ◦ MSE: 154.23
        ◦ R² Score: 0.89
These results indicate that the Random Forest Regressor model provided a good fit for the dataset and was able to estimate project costs with reasonable accuracy.



Predicted Project Cost Example:
    • Based on user input:
        ◦ Project Size: 2000 function points	
        ◦ Project Duration: 18 months   		
        ◦ Team Size: 12
        ◦ Complexity: 7
        ◦ Reliability: 4
        ◦ Database Size: 500 GB		
        ◦ Team Cohesion: 4				
        ◦ Developer Experience: 4
        ◦ Software Tools: 3
      The predicted project cost was $191,834.76.

Conclusion:
    • The Random Forest Regressor provided better results than Linear Regression, and the estimated project cost now matches the sum of the individual contributions.
    • The tool can accurately estimate the cost of software projects, making it a valuable resource for project managers.
=========================================================================
