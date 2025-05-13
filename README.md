```
.
├── .gitignore
├── README.md
├── buildspec.yml          # AWS CodeBuild specification file
├── template.json          # AWS SAM or CloudFormation template defining resources
├── stepfunctions          # Directory for Step Function definitions
│   └── my_state_machine.asl.json # Amazon States Language definition for your Step Function
└── src                    # Directory for application source code (e.g., Lambda functions)
├── transform_data_lambda     # Directory for the 'TransformData' Lambda function
│   ├── vpp-transformation.py              # Lambda handler code (Python example)
├── generate_basic_dashboard_lambda # Directory for 'GenerateBasicDashboard' Lambda
│   └── vpp-dashboard-generator.py    # Lambda handler code (Python example)
└── generate_mi_dashboard_lambda    # Directory for 'GenerateMIDashboard' Lambda
├── lambda_function.py   # Lambda handler code (Another Python example)
└── requirements.txt
```
**Explanation of the key files and folders:**

* `.gitignore`, `README.md`: Standard project files for version control and documentation.
* `buildspec.yml`: This file is used by AWS CodeBuild. It contains the commands to package your application (e.g., using `sam package`) and prepare the artifacts for deployment.
* `template.json`: This is your AWS SAM or CloudFormation template. It defines all the AWS resources in your application, including:
    * The AWS Step Function (`AWS::StepFunctions::StateMachine`) pointing to the `my_state_machine.asl.json` file.
    * The AWS Lambda functions (`AWS::Serverless::Function` for SAM or `AWS::Lambda::Function` for CloudFormation) pointing to the code in the `src` subdirectories.
    * IAM Roles and policies needed by the Step Function and Lambda functions.
    * Any other resources (like S3 buckets, DynamoDB tables, etc.) your application uses.
* `stepfunctions/`: A directory to hold the JSON or YAML definition file for your Step Function written in Amazon States Language. Your `template.yaml` will reference this file.
* `src/`: A directory containing the code for your different Lambda functions. Each Lambda typically has its own subdirectory containing the handler code and any dependencies required (like `requirements.txt` for Python or `package.json` for Node.js).
