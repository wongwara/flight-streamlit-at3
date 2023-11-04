# Function to convert a duration string to seconds
import re
def convert_duration_to_seconds(duration):
    match = re.match(r'PT(\d+)H(\d+)M', duration)
    
    if match:
        hours = int(match.group(1))
        minutes = int(match.group(2))
        total_seconds = hours * 3600 + minutes * 60
        return total_seconds
    else:
        return None

# Function to split and sum the values
def split_and_sum(segment_duration):
    segments = segment_duration.split('||')
    return sum(map(int, segments))

# Function to process the code list
def process_code_list(code_list):
    if len(code_list) != 1:
        code_list = code_list[1:] 
    return code_list

# Define a function to split the string and create a list
def split_duration(segment):
    return [int(value) for value in re.split(r'\|\|', segment)]

# Define a function to split the description and create a list
def split_description(segment):
    if segment and isinstance(segment, str):
        return [description.strip() for description in re.split(r'\|\|', segment) if description]
    else:
        return []

def get_first_element(x):
    return x[0] if len(x) > 0 else None

def get_last_element(x):
    return x[-1] if len(x) > 0 else None

