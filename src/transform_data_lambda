import boto3
import pandas as pd
from io import StringIO
import os
from datetime import datetime, timezone

def lambda_handler(event, context):
    s3 = boto3.client('s3')
    input_bucket = os.environ.get('S3_BUCKET_NAME')
    input_key = event['key']
    output_key = 'transformed_charge_data.csv'

    try:
        response = s3.get_object(Bucket=input_bucket, Key=input_key)
        csv_content = response['Body'].read().decode('utf-8')
        df = pd.read_csv(StringIO(csv_content))

        timestamp_column = 'CHARGE_START_TIME_AT'
        energy_column = 'CHARGE_WATT_HOUR'
        duration_column = 'CHARGE_DURATION_MINS'
        stop_time_column = 'CHARGE_STOP_TIME_AT'

        # Ensure timestamp columns are datetime objects
        df[timestamp_column] = pd.to_datetime(df[timestamp_column])
        df[stop_time_column] = pd.to_datetime(df[stop_time_column])

        # Filter out records with CHARGE_START_TIME_AT in the future
        now_utc = datetime.now(timezone.utc)
        df_filtered = df[df[timestamp_column] <= now_utc]

        # Calculate a flag for records where start and stop times are the same but duration or energy is non-zero
        df_filtered['start_stop_same_non_zero'] = (
            (df_filtered[timestamp_column] == df_filtered[stop_time_column]) &
            ((df_filtered[duration_column] > 0) | (df_filtered[energy_column] > 0))
        ).astype(int)

        # Correct charge duration to zero if start and stop times are the same
        df_filtered['charge_duration_mins_corrected'] = df_filtered.apply(
            lambda row: 0 if row[timestamp_column] == row[stop_time_column]
            else row[duration_column],
            axis=1
        )

        # Calculate charge duration in hours
        df_filtered['charge_duration_hours'] = df_filtered['charge_duration_mins_corrected'] / 60

        # Calculate hourly watt-hour per station charge, handling zero duration to avoid division by zero
        df_filtered['hourly_watt_hour_per_station_charge'] = df_filtered.apply(
            lambda row: 0 if row['charge_duration_hours'] == 0
            else row[energy_column] / row['charge_duration_hours'],
            axis=1
        )

        # Create a flag for missing CHARGE_STOP_TIME_AT
        df_filtered['is_stop_time_missing'] = df_filtered[stop_time_column].isnull().astype(int)

        # Select the columns for the transformed data
        transformed_df = df_filtered[[
            'STATION_ID',
            'CHARGE_ID',
            timestamp_column,
            stop_time_column,
            energy_column,
            'charge_duration_mins_corrected',
            'charge_duration_hours',
            'hourly_watt_hour_per_station_charge',
            'is_stop_time_missing',
            'start_stop_same_non_zero'
        ]]

        # Save the transformed data to S3
        csv_buffer = StringIO()
        transformed_df.to_csv(csv_buffer, index=False)
        transformed_data_csv = csv_buffer.getvalue()

        s3.put_object(Bucket=input_bucket, Key=output_key, Body=transformed_data_csv)

        print(f"Successfully saved transformed data to s3://{input_bucket}/{output_key}")

        return {
            'statusCode': 200,
            'body': f'Successfully transformed data and saved to s3://{input_bucket}/{output_key}'
        }

    except Exception as e:
        print(f"Error processing S3 object: {e}")
        return {
            'statusCode': 500,
            'body': f'Error processing S3 object: {e}'
        }
