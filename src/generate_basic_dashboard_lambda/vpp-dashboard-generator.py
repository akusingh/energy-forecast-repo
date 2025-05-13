import boto3
import pandas as pd
from io import StringIO
import os
import json
from datetime import timezone

def lambda_handler(event, context):
    s3 = boto3.client('s3')
    bucket_name = os.environ.get('S3_BUCKET_NAME')
    key = 'transformed_charge_data.csv'

    print(f"Attempting to read from bucket: {bucket_name}, key: {key}")

    try:
        response = s3.get_object(Bucket=bucket_name, Key=key)
        print(f"S3 GetObject response: {response}")
        csv_content = response['Body'].read().decode('utf-8')
        df = pd.read_csv(StringIO(csv_content))

        print(f"Columns in DataFrame (Dashboard Lambda): {df.columns.tolist()}")
        print(f"Data type of CHARGE_WATT_HOUR column: {df['CHARGE_WATT_HOUR'].dtype}")
        print(f"Sample of CHARGE_WATT_HOUR values (before resample):\n{df['CHARGE_WATT_HOUR'].head()}")

        # Explicitly convert CHARGE_START_TIME_AT to datetime and set as index
        df['CHARGE_START_TIME_AT'] = pd.to_datetime(df['CHARGE_START_TIME_AT'])
        df.set_index('CHARGE_START_TIME_AT', inplace=True)
        df = df.sort_index()

        # Get the actual date range from the data
        min_date = df.index.min()
        max_date = df.index.max()
        print(f"Date range in data: {min_date} to {max_date}")

        # Explicitly resample the CHARGE_WATT_HOUR column and sum
        hourly_total = df['CHARGE_WATT_HOUR'].resample('h').sum()

        # *** ADD THIS LOGGING ***
        print("Sample of hourly_total (after resample, before reindex):\n", hourly_total.head())

        # Reindex to ensure all hours are present, creating the index from the resampled data
        hourly_index = pd.date_range(start=hourly_total.index.min(), end=hourly_total.index.max(), freq='h', tz=hourly_total.index.tz)
        hourly_total = hourly_total.reindex(hourly_index, fill_value=0)

        # Get the actual date range of the aggregated data
        min_date = hourly_total.index.min()
        max_date = hourly_total.index.max()
        print(f"Date range in aggregated data: {min_date} to {max_date}")

        # *** ADD THESE LOGGING STATEMENTS ***
        print("hourly_total.index:\n", hourly_total.index)
        print("hourly_total.values:\n", hourly_total.values)

        # Prepare data for Chart.js
        labels = hourly_total.index.strftime('%Y-%m-%d %H:%M:%S%z').tolist()
        data = hourly_total.values.tolist()
        print(f"Sample of data being passed to Chart.js: {data[:10]}")

        # Generate basic HTML dashboard with Chart.js
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Energy Consumption Dashboard</title>
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        </head>
        <body>
            <h1>Total Hourly Energy Consumption</h1>
            <canvas id="energyChart"></canvas>
            <script>
                const data = {{
                    labels: {json.dumps(labels)},
                    datasets: [{{
                        label: 'Total Watt-Hours',
                        data: {json.dumps(data)},
                        borderColor: 'rgb(54, 162, 235)',
                        backgroundColor: 'rgba(54, 162, 235, 0.2)',
                        tension: 0.1
                    }}]
                }};

                const config = {{
                    type: 'line',
                    data: data,
                    options: {{
                        scales: {{
                            y: {{
                                beginAtZero: true
                            }}
                        }}
                    }}
                }};

                const energyChart = new Chart(
                    document.getElementById('energyChart'),
                    config
                );
            </script>
        </body>
        </html>
        """

        return {
            'statusCode': 200,
            'headers': {'Content-Type': 'text/html'},
            'body': html_content,
        }

    except Exception as e:
        print(f"Error in energy-dashboard-generator: {e}")
        return {
            'statusCode': 500,
            'headers': {'Content-Type': 'text/plain'},
            'body': f'Error: {str(e)}',
        }
