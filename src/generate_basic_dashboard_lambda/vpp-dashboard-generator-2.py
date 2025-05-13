import boto3
import pandas as pd
from io import StringIO
import os
import json
from datetime import timezone
from botocore.exceptions import ClientError
import numpy as np # Import numpy to check types

def lambda_handler(event, context):
    """
    AWS Lambda function to read energy consumption data from S3,
    process it, and generate an HTML dashboard with Chart.js and Flatpickr
    for interactive filtering.
    """
    s3 = boto3.client('s3')
    # Get the S3 bucket name from environment variables
    bucket_name = os.environ.get('S3_BUCKET_NAME')
    # Define the key (file path) for the CSV file in S3
    key = 'transformed_charge_data.csv'

    print(f"Attempting to read from bucket: {bucket_name}, key: {key}")

    try:
        # Fetch the CSV file from S3
        response = s3.get_object(Bucket=bucket_name, Key=key)
        print(f"S3 GetObject successful.")
        # Read the content and decode it from bytes to a UTF-8 string
        csv_content = response['Body'].read().decode('utf-8')
        # Use StringIO to treat the string content as a file for pandas
        df = pd.read_csv(StringIO(csv_content))

        print(f"DataFrame loaded. Columns: {df.columns.tolist()}")
        # Check if the DataFrame is empty
        if df.empty:
            print("DataFrame is empty. No data to process.")
            return {
                'statusCode': 200,
                'headers': {'Content-Type': 'text/html'},
                'body': '<h1>No data available to generate dashboard.</h1>',
            }

        # Ensure required columns exist
        required_columns = ['CHARGE_START_TIME_AT', 'CHARGE_WATT_HOUR', 'STATION_ID']
        if not all(col in df.columns for col in required_columns):
             missing = [col for col in required_columns if col not in df.columns]
             error_msg = f"Missing required columns: {', '.join(missing)}"
             print(error_msg)
             return {
                'statusCode': 500,
                'headers': {'Content-Type': 'text/plain'},
                'body': f'Error: {error_msg}',
            }

        # Explicitly convert CHARGE_START_TIME_AT to datetime with UTC timezone
        # Use errors='coerce' to turn unparseable dates into NaT (Not a Time)
        df['CHARGE_START_TIME_AT'] = pd.to_datetime(df['CHARGE_START_TIME_AT'], utc=True, errors='coerce')

        # Drop rows where datetime conversion failed
        df.dropna(subset=['CHARGE_START_TIME_AT'], inplace=True)

        # Ensure CHARGE_WATT_HOUR is numeric, coercing errors to NaN and filling with 0
        df['CHARGE_WATT_HOUR'] = pd.to_numeric(df['CHARGE_WATT_HOUR'], errors='coerce').fillna(0)

        # Set the datetime column as the index and sort
        df.set_index('CHARGE_START_TIME_AT', inplace=True)
        df = df.sort_index()

        print(f"Data processed. Index set to CHARGE_START_TIME_AT.")
        print(f"Sample data head:\n{df.head()}")
        print(f"Data types after processing:\n{df.dtypes}")


        # --- Data Aggregation for Dashboard ---
        # Aggregate data by month for ALL stations
        # Group by Station ID and then resample the time index by month ('M')
        # Calculate the sum of 'CHARGE_WATT_HOUR' for each month per station
        # Use .reset_index() to turn the MultiIndex into columns for easier processing
        all_monthly_data_df = df.groupby('STATION_ID').resample('M')['CHARGE_WATT_HOUR'].sum().fillna(0).reset_index()
        # Rename the time column for clarity
        all_monthly_data_df.rename(columns={'CHARGE_START_TIME_AT': 'Month'}, inplace=True)

        print(f"Aggregated monthly data head:\n{all_monthly_data_df.head()}")


        # Structure the data for JavaScript
        # Create a dictionary where keys are station IDs (as strings)
        # and values are lists of monthly totals (as standard floats)
        stations_data_for_js = {}
        # Get all unique station IDs and convert them to strings
        all_station_ids = [str(sid) for sid in all_monthly_data_df['STATION_ID'].unique().tolist()]

        # Get all unique month labels (YYYY-MM format) across the *entire* aggregated dataset
        all_month_labels = all_monthly_data_df['Month'].dt.strftime('%Y-%m').unique().tolist()

        # Ensure all_month_labels are sorted chronologically
        all_month_labels.sort()


        for station_id_str in all_station_ids:
            # Filter the aggregated DataFrame for the current station (using the string ID)
            # Need to handle potential type differences if STATION_ID was numeric in CSV
            # Ensure lookup works by comparing string representation if needed
            station_df = all_monthly_data_df[all_monthly_data_df['STATION_ID'].astype(str) == station_id_str].copy()

            # Create a Series with all months in the dataset range, filled with 0, then update with station data
            station_monthly_series = pd.Series(0.0, index=all_month_labels)
            if not station_df.empty:
                 # Map the monthly sums from the station_df to the full list of month labels
                 station_df['MonthStr'] = station_df['Month'].dt.strftime('%Y-%m')
                 station_monthly_series.update(station_df.set_index('MonthStr')['CHARGE_WATT_HOUR'])

            # Convert the Series values to a list of standard Python floats
            stations_data_for_js[station_id_str] = station_monthly_series.tolist()


        print(f"Aggregated data structured for JS for {len(all_station_ids)} stations over {len(all_month_labels)} months.")

        # --- Debugging Print Statements ---
        print(f"Data types before JSON serialization:")
        print(f"Type of all_month_labels: {type(all_month_labels)}, first element type: {type(all_month_labels[0]) if all_month_labels else 'N/A'}")
        print(f"Type of all_station_ids: {type(all_station_ids)}, first element type: {type(all_station_ids[0]) if all_station_ids else 'N/A'}")
        print(f"Type of stations_data_for_js: {type(stations_data_for_js)}")
        if stations_data_for_js:
            first_station_key = list(stations_data_for_js.keys())[0]
            print(f"Type of first station key: {type(first_station_key)}")
            first_station_data_list = stations_data_for_js[first_station_key]
            print(f"Type of first station data list: {type(first_station_data_list)}")
            if first_station_data_list:
                 print(f"Type of first element in first station data list: {type(first_station_data_list[0])}")
        # --- End Debugging Print Statements ---


        # Prepare data to be embedded in the HTML script tag
        # json.dumps converts Python objects to JSON strings.
        # The data types should now be standard Python types (list of strings, dict of lists of floats, list of strings)
        js_labels = json.dumps(all_month_labels)
        js_stations_data = json.dumps(stations_data_for_js)
        js_station_ids = json.dumps(all_station_ids)


        # Generate HTML dashboard with Chart.js and Flatpickr
        # Include options for all stations in the dropdown
        station_options_html = ''.join([f'<option value="{station_id}">{station_id}</option>' for station_id in all_station_ids])

        # Start of the HTML content string using an f-string
        # Ensure all curly braces within the HTML/CSS/JS are escaped by doubling them {{ }}
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Energy Consumption Dashboard</title>
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
            <script src="https://cdn.jsdelivr.net/npm/date-fns"></script>
            <script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns"></script>
             <script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-datalabels@2.0.0"></script>
            <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/flatpickr/dist/flatpickr.min.css">
            <script src="https://cdn.jsdelivr.net/npm/flatpickr"></script>
            <style>
                body {{
                    font-family: sans-serif;
                    margin: 20px;
                    background-color: #f4f4f4;
                }}
                h1 {{
                    text-align: center;
                    color: #333;
                }}
                #controls-container {{
                    display: flex;
                    justify-content: center;
                    gap: 20px; /* Space between control groups */
                    margin-bottom: 20px;
                    flex-wrap: wrap; /* Allow controls to wrap on smaller screens */
                }}
                .control-group {{
                    display: flex;
                    align-items: center;
                    background-color: #fff;
                    padding: 10px 15px;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
                }}
                .control-group label {{
                    margin-right: 10px;
                    font-weight: bold;
                    color: #555;
                }}
                .control-group select,
                .control-group input[type="text"] {{
                    padding: 8px;
                    border: 1px solid #ccc;
                    border-radius: 4px;
                    font-size: 1rem;
                }}
                 #date-range {{
                    width: 250px; /* Adjust width as needed */
                 }}
                 #station-filter {{
                    width: 150px; /* Adjust width as needed */
                 }}

                #chart-container {{
                    background-color: #fff;
                    padding: 20px;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
                    max-width: 90%; /* Use max-width for responsiveness */
                    margin: 0 auto; /* Center the container */
                    position: relative; /* Needed for potential loading indicator */
                }}
                #energyChart {{
                    width: 100% !important; /* Override default canvas size */
                    height: 500px !important; /* Set a fixed height or use aspect ratio */
                }}
                 #no-data-message {{
                    text-align: center;
                    color: #888;
                    font-size: 1.2rem;
                    margin-top: 50px;
                 }}
            </style>
        </head>
        <body>
            <h1>Energy Consumption Dashboard</h1>

            <div id="controls-container">
                <div class="control-group">
                    <label for="date-range">Select Date Range:</label>
                    <input type="text" id="date-range" placeholder="Select Date Range">
                </div>

                <div class="control-group">
                    <label for="station-filter">Select Station:</label>
                    <select id="station-filter">
                        <option value="all">All Stations</option>
                        {station_options_html}
                    </select>
                </div>
            </div>

            <div id="chart-container">
                 <canvas id="energyChart"></canvas>
                 <div id="no-data-message" style="display: none;">No data available for the selected filters.</div>
            </div>


            <script>
                // Data passed from the Lambda function (Python)
                // These variables hold the *entire* monthly aggregated data for all stations
                const allMonthLabels = {js_labels}; // e.g., ["2023-11", "2023-12", ...]
                const allStationsData = {js_stations_data}; // e.g., {{"station1": [100.0, 120.0, ...], "station2": [50.0, 60.0, ...], ...}}
                const allStationIds = {js_station_ids}; // e.g., ["station1", "station2", ...]

                let energyChart;
                const chartCtx = document.getElementById('energyChart').getContext('2d');
                const noDataMessage = document.getElementById('no-data-message');

                // Function to update the chart based on selected filters
                function updateChart() {{
                    const dateRangeInput = document.getElementById('date-range');
                    const stationFilter = document.getElementById('station-filter');

                    // Flatpickr uses ' to ' as the default separator for range mode
                    const selectedDateRange = dateRangeInput.value.split(' to ');
                    const selectedStation = stationFilter.value;

                    let startDate = null;
                    let endDate = null;

                    if (selectedDateRange.length === 2 && selectedDateRange[0] && selectedDateRange[1]) {{
                        // Parse selected dates as Date objects (Flatpickr provides YYYY-MM-DD)
                        startDate = new Date(selectedDateRange[0]);
                        endDate = new Date(selectedDateRange[1]);
                         // Adjust end date to include the whole day
                        endDate.setHours(23, 59, 59, 999);
                    }}

                    let filteredLabels = [];
                    let filteredData = [];

                    // Iterate through all available month labels from the full dataset
                    for (let i = 0; i < allMonthLabels.length; i++) {{
                        const monthLabel = allMonthLabels[i];
                        // Create a Date object for the first day of the month for comparison
                        const monthDate = new Date(monthLabel + '-01');

                        // Apply date range filter if dates are selected
                        if (startDate && endDate) {{
                            // Check if the month's date falls within the selected range
                            // Compare monthDate (start of month) with startDate and endDate
                            if (monthDate < startDate || monthDate > endDate) {{
                                continue; // Skip this month if it's outside the selected range
                            }}
                        }}

                        // If the month is within the date range (or no range is selected)
                        filteredLabels.push(monthLabel);

                        if (selectedStation === 'all') {{
                            // Calculate the sum of watt-hours for this month across ALL stations
                            let monthlyTotal = 0;
                            allStationIds.forEach(stationId => {{
                                // Ensure the index 'i' exists in the data array for this station
                                if (allStationsData[stationId] && allStationsData[stationId].length > i) {{
                                    monthlyTotal += allStationsData[stationId][i];
                                }}
                            }});
                            filteredData.push(monthlyTotal);
                        }} else {{
                            // Get the watt-hours data for the specific selected station for this month
                            if (allStationsData[selectedStation] && allStationsData[selectedStation].length > i) {{
                                filteredData.push(allStationsData[selectedStation][i]);
                            }} else {{
                                // If data for this station/month is missing (shouldn't happen with the Python logic), use 0
                                filteredData.push(0);
                            }}
                        }}
                    }}

                    // Update chart data
                    if (energyChart) {{
                        energyChart.data.labels = filteredLabels;
                        energyChart.data.datasets[0].data = filteredData;
                        // Update the dataset label based on selection
                        // Escape the curly braces in the JavaScript template literal for the Python f-string
                        energyChart.data.datasets[0].label = selectedStation === 'all' ? 'Total Watt-Hours (All Stations)' : `Total Watt-Hours (Station ${{selectedStation}})`;

                        // Use a different color for the specific station vs all stations
                         if (selectedStation === 'all') {{
                            energyChart.data.datasets[0].backgroundColor = 'rgba(54, 162, 235, 0.6)'; // Blue for all
                            energyChart.data.datasets[0].borderColor = 'rgba(54, 162, 235, 1)';
                         }} else {{
                             // You could add logic here to pick different colors for different stations
                             energyChart.data.datasets[0].backgroundColor = 'rgba(255, 99, 132, 0.6)'; // Red for single station
                             energyChart.data.datasets[0].borderColor = 'rgba(255, 99, 132, 1)';
                         }}


                        energyChart.update(); // Redraw the chart
                    }}

                    // Show or hide the "No data" message based on the filtered data
                    const hasDataToShow = filteredData.length > 0 && filteredData.some(value => value > 0);
                    if (hasDataToShow) {{
                         noDataMessage.style.display = 'none';
                         chartCtx.canvas.style.display = 'block'; // Show canvas
                     }} else {{
                         noDataMessage.style.display = 'block';
                         chartCtx.canvas.style.display = 'none'; // Hide canvas
                     }}
                }}

                // Initialize Flatpickr date range picker on the input element
                const dateRangePicker = flatpickr("#date-range", {{
                    mode: "range", // Enable range selection
                    dateFormat: "Y-m-d", // Set the date format
                    // Callback function when the date picker is closed
                    onClose: function(selectedDates, dateStr, instance) {{
                        // selectedDates is an array of Date objects [startDate, endDate]
                        // dateStr is the string representation of the selected range
                        // Trigger the chart update function
                        updateChart();
                    }}
                }});

                // Add event listener to the station filter dropdown
                document.getElementById('station-filter').addEventListener('change', function() {{
                    // Trigger the chart update function when the selected station changes
                    updateChart();
                }});

                // This function runs when the entire page (including external resources like scripts) has loaded
                window.onload = function() {{
                    // Register the ChartDataLabels plugin with Chart.js
                    Chart.register(ChartDataLabels);

                    // Initial chart configuration
                    const initialChartConfig = {{
                        type: 'bar', // Define the chart type (e.g., 'bar', 'line')
                        data: {{
                            labels: [], // Initial labels are empty, will be populated by updateChart
                            datasets: [{{
                                label: 'Total Watt-Hours', // Initial label, will be updated
                                data: [], // Initial data is empty, will be populated by updateChart
                                backgroundColor: 'rgba(54, 162, 235, 0.6)', // Default bar color
                                borderColor: 'rgba(54, 162, 235, 1)', // Default border color
                                borderWidth: 1
                            }}]
                        }},
                        options: {{
                            responsive: true, // Chart will resize with the container
                            maintainAspectRatio: false, // Allow CSS to control height
                            animation: false, // Disable initial animation for potentially faster rendering
                            plugins: {{
                                legend: {{
                                    display: true, // Show the legend
                                    position: 'top', // Position the legend at the top
                                }},
                                datalabels: {{
                                    anchor: 'end', // Position label at the end of the bar
                                    align: 'top', // Align label to the top of the bar
                                    offset: 5, // Distance from the bar
                                    color: '#333', // Label text color
                                    font: {{
                                        weight: 'bold',
                                        size: 10 // Font size for labels
                                    }},
                                    formatter: function(value, context) {{
                                        // Custom formatter function for data labels
                                        // You can add units, round values, etc.
                                        return value > 0 ? value.toFixed(0) : ''; // Show value if > 0, rounded to integer
                                    }}
                                }}
                            }},
                            scales: {{
                                x: {{
                                    title: {{
                                        display: true,
                                        text: 'Month' // X-axis title
                                    }}
                                }},
                                y: {{
                                    beginAtZero: true, // Start Y-axis at zero
                                     title: {{
                                        display: true,
                                        text: 'Watt-Hours' // Y-axis title
                                    }}
                                }}
                            }}
                        }},
                        plugins: [ChartDataLabels] // Explicitly add the datalabels plugin to this chart instance
                    }};

                    // Create the new Chart.js instance
                    energyChart = new Chart(chartCtx, initialChartConfig);

                    // Perform the initial chart update to display data based on default selections (all stations, no date filter initially)
                    updateChart();
                }};
            </script>
        </body>
        </html>
        """

        # Return the HTML content as the Lambda response
        return {
            'statusCode': 200,
            'headers': {'Content-Type': 'text/html'},
            'body': html_content,
        }

    except ClientError as e:
        # Handle S3 access errors
        print(f"Error accessing S3: {e}")
        return {
            'statusCode': 500,
            'headers': {'Content-Type': 'text/plain'},
            'body': f'Error accessing S3: {str(e)}',
        }
    except pd.errors.EmptyDataError as e:
        # Handle case where CSV is empty but exists
        print(f"Error reading CSV data: {e}")
        return {
            'statusCode': 200, # Return 200 as it's not a server error, just no data
            'headers': {'Content-Type': 'text/html'},
            'body': '<h1>No data found in the CSV file.</h1>',
        }
    except Exception as e:
        # Catch any other unexpected errors
        print(f"An unexpected error occurred: {e}")
        # Print the full traceback for debugging
        import traceback
        traceback.print_exc()
        return {
            'statusCode': 500,
            'headers': {'Content-Type': 'text/plain'},
            'body': f'An unexpected error occurred: {str(e)}',
        }
