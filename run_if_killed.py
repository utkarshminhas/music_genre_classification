'''runs a python file again and again indefinitely if that python file happens to kill itself and it processes at random points of time'''

import subprocess

# Path to the Python file to run
python_file = '4_process_songs.py'

# # Run the Python file indefinitely
# while True:
#     try:
#         result = subprocess.run(['python', python_file], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
#         print(f"Output: {result.stdout}")
#     except subprocess.CalledProcessError as e:
#         print(f"Error: {e.stderr}")

# Run the Python file indefinitely
while True:
    try:
       
       # Open a subprocess to run the Python file in the background
       process = subprocess.Popen(['python', python_file], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
       # Print the output of the Python file to the terminal
       for line in process.stdout:
          print(line, end='')
    except subprocess.CalledProcessError as e:
        print(f"Error: {e.stderr}")