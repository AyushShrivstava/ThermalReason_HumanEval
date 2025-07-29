import streamlit as st
import os
import random
import pandas as pd
import uuid
import time
import ast
from datetime import datetime
from PIL import Image, ImageDraw

# Set page config
st.set_page_config(page_title="Human Benchmarking", layout="centered")

# Function to draw arrow on image
def draw_arrow_on_image(image_path, x, y):
    """Draw an arrow pointing to specific coordinates on the image"""
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    
    # Arrow properties (made bigger)
    arrow_color = "red"
    arrow_width = 12
    arrow_length = 120
    
    # Calculate arrow points (pointing downward to the target)
    start_x = x
    start_y = max(0, y - arrow_length)
    end_x = x
    end_y = y
    
    # Draw arrowhead first to get its size
    arrow_head_size = 35
    
    # Draw main arrow line - stop before the arrowhead to avoid rectangle effect
    line_end_y = end_y - arrow_head_size + 10  # Stop 10 pixels before arrowhead base
    draw.line([(start_x, start_y), (end_x, line_end_y)], fill=arrow_color, width=arrow_width)
    
    # Draw arrowhead (made bigger) - filled triangle
    head_points = [
        (end_x, end_y),
        (end_x - arrow_head_size//2, end_y - arrow_head_size),
        (end_x + arrow_head_size//2, end_y - arrow_head_size)
    ]
    draw.polygon(head_points, fill=arrow_color, outline=arrow_color)
    
    return image

# Function to get all images from multiple directories and shuffle them
def get_all_images(image_dirs):
    if isinstance(image_dirs, str):
        image_dirs = [image_dirs]
    
    all_images = []
    for image_dir in image_dirs:
        if os.path.exists(image_dir):
            image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            # Add directory prefix to maintain uniqueness
            dir_name = os.path.basename(image_dir)
            prefixed_files = [(os.path.join(image_dir, f), f) for f in image_files]
            all_images.extend(prefixed_files)
    
    # Sort by filename for consistency, then shuffle for randomness
    all_images = sorted(all_images, key=lambda x: x[1])
    random.shuffle(all_images)
    return all_images

# Function to load thermal reasoning data from CSV
def get_thermal_reasoning_data(csv_file, task_type):
    """Load and shuffle thermal reasoning task data from CSV files"""
    csv_path = f"data/task4/{csv_file}"
    if not os.path.exists(csv_path):
        return []
    
    df = pd.read_csv(csv_path)
    
    if task_type == "thermal_reasoning_1":
        # For double_sampled.csv: comparison tasks
        task_data = []
        for _, row in df.iterrows():
            image_path = os.path.join("data/thermEval_data", row['image_name'])
            task_data.append({
                'image_path': image_path,
                'image_name': row['image_name'],
                'body_part': row['body_part'],
                'correct_answer': row['correct_answer'],
                'question': f"Which person has the hotter {row['body_part']}?"
            })
    else:
        # For single_sampled.csv: ordering tasks
        task_data = []
        for _, row in df.iterrows():
            image_path = os.path.join("data/thermEval_data", row['image_name'])
            # Parse the correct_order string as a list
            correct_order = ast.literal_eval(row['correct_order'])
            task_data.append({
                'image_path': image_path,
                'image_name': row['image_name'],
                'correct_order': correct_order,
                'question': f"Arrange these body parts from coldest to hottest: {', '.join(correct_order)}"
            })
    
    # Shuffle the data
    random.shuffle(task_data)
    return task_data

# Function to load temperature estimation data from CSV
def get_temperature_estimation_data(csv_file, task_type):
    """Load and shuffle temperature estimation task data from CSV files"""
    csv_path = f"data/task5/{csv_file}"
    if not os.path.exists(csv_path):
        return []
    
    df = pd.read_csv(csv_path)
    
    if task_type == "arrow_temp_estimation":
        # For arrow_sampled.csv: arrow-based temperature estimation
        task_data = []
        for _, row in df.iterrows():
            image_path = os.path.join("data/thermEval_data", row['image_name'])
            task_data.append({
                'image_path': image_path,
                'image_name': row['image_name'],
                'x': row['x'],
                'y': row['y'],
                'actual_temperature': row['temperature'],
                'question': f"Estimate the temperature at the arrow location (in °C)"
            })
    elif task_type == "semantic_temp_estimation":
        # For region_sampled.csv: semantic temperature estimation (single instance)
        task_data = []
        for _, row in df.iterrows():
            if row['instance'] == 'single':
                image_path = os.path.join("data/thermEval_data", row['image_name'])
                task_data.append({
                    'image_path': image_path,
                    'image_name': row['image_name'],
                    'body_part': row['body_part'],
                    'actual_temperature': row['temperature'],
                    'question': f"Estimate the temperature of the {row['body_part']} (in °C)"
                })
    elif task_type == "comparative_temp_estimation":
        # For region_sampled.csv: comparative temperature estimation (double instance)
        task_data = []
        for _, row in df.iterrows():
            if row['instance'] == 'double':
                image_path = os.path.join("data/thermEval_data", row['image_name'])
                side = row['evaluate']  # 'left' or 'right'
                task_data.append({
                    'image_path': image_path,
                    'image_name': row['image_name'],
                    'body_part': row['body_part'],
                    'side': side,
                    'actual_temperature': row['temperature'],
                    'question': f"Estimate the temperature of the {side} person's {row['body_part']} (in °C)"
                })
    
    # Shuffle the data
    random.shuffle(task_data)
    return task_data

# Function to save results to CSV
def save_results(user_id, user_info, task, image_name, answer, response_time=None):
    results_file = "human_evaluation_results.csv"
    
    # Create new row
    new_row = {
        'user_id': user_id,
        'name': user_info['name'],
        'age': user_info['age'],
        'occupation': user_info['occupation'],
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'task': task,
        'image_name': image_name,
        'answer': answer,
        'response_time_milliseconds': response_time
    }
    
    # Check if file exists
    if os.path.exists(results_file):
        df = pd.read_csv(results_file)
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    else:
        df = pd.DataFrame([new_row])
    
    df.to_csv(results_file, index=False)

# Initialize session state
if 'user_id' not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())[:8]
if 'user_info_collected' not in st.session_state:
    st.session_state.user_info_collected = False
if 'user_info' not in st.session_state:
    st.session_state.user_info = {}
if 'task_progress' not in st.session_state:
    st.session_state.task_progress = {}
if 'current_task' not in st.session_state:
    st.session_state.current_task = None
if 'image_start_time' not in st.session_state:
    st.session_state.image_start_time = None

# Title
st.title("Human Baseline Benchmarking")

# Developer-only admin section with password protection
st.sidebar.markdown("---")
st.sidebar.markdown("**Developer Section**")

# Initialize admin access state
if 'admin_access' not in st.session_state:
    st.session_state.admin_access = False

# Admin login
if not st.session_state.admin_access:
    admin_password = st.sidebar.text_input("Developer Password:", type="password", key="admin_password")
    if st.sidebar.button("Access Admin Panel"):
        # Set your developer password here
        DEVELOPER_PASSWORD = "thermal2025"  # Change this to your preferred password
        
        if admin_password == DEVELOPER_PASSWORD:
            st.session_state.admin_access = True
            st.sidebar.success("Admin access granted!")
            st.rerun()
        else:
            st.sidebar.error("Incorrect password")
else:
    # Admin panel - only visible when authenticated
    st.sidebar.success("🔓 Admin Panel Active")
    
    if st.sidebar.button("Download Results CSV"):
        results_file = "human_evaluation_results.csv"
        if os.path.exists(results_file):
            df = pd.read_csv(results_file)
            csv_data = df.to_csv(index=False)
            st.sidebar.download_button(
                label="Download CSV File",
                data=csv_data,
                file_name=f"human_evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
            st.sidebar.success(f"Ready to download! {len(df)} records found.")
        else:
            st.sidebar.warning("No results file found yet.")
    
    # Display current results count (admin only)
    results_file = "human_evaluation_results.csv"
    if os.path.exists(results_file):
        df = pd.read_csv(results_file)
        st.sidebar.info(f"Total responses: {len(df)}")
        unique_users = df['user_id'].nunique() if 'user_id' in df.columns else 0
        st.sidebar.info(f"Unique users: {unique_users}")
    
    if st.sidebar.button("Logout"):
        st.session_state.admin_access = False
        st.rerun()

# Collect user information first
if not st.session_state.user_info_collected:
    st.header("Welcome! Please provide your information:")
    
    with st.form("user_info_form"):
        name = st.text_input("Name:")
        age = st.number_input("Age:", min_value=1, max_value=100, step=1)
        occupation = st.text_input("Occupation:")
        submitted = st.form_submit_button("Start Evaluation")
        
        if submitted and name and age and occupation:
            st.session_state.user_info = {
                'name': name,
                'age': age,
                'occupation': occupation
            }
            st.session_state.user_info_collected = True
            st.rerun()
    
    if not (name and age and occupation) and submitted:
        st.error("Please fill in all fields.")
    
    st.info(f"Your unique ID: {st.session_state.user_id}")
    st.stop()

# Sidebar for task selection
task = st.sidebar.selectbox("Choose a Task", [
    "Modality identification", 
    "Human Counting", 
    "Thermal Reasoning 1", 
    "Thermal Reasoning 2",
    "Arrow based temp estimation",
    "Semantic Temperature Estimation",
    "Comparative Temperature Estimation"
])

# Load images for the selected task
if task == "Modality identification":
    # Combine images from task1 and task2 directories
    task_dirs = ["data/task1", "data/task2"]
    task_type = "thermal_classification"
elif task == "Human Counting":
    task_dirs = ["data/task3"]
    task_type = "people_counting"
elif task == "Thermal Reasoning 1":
    task_type = "thermal_reasoning_1"
    task_dirs = None  # Will use CSV data instead
elif task == "Thermal Reasoning 2":
    task_type = "thermal_reasoning_2"
    task_dirs = None  # Will use CSV data instead
elif task == "Arrow based temp estimation":
    task_type = "arrow_temp_estimation"
    task_dirs = None  # Will use CSV data instead
elif task == "Semantic Temperature Estimation":
    task_type = "semantic_temp_estimation"
    task_dirs = None  # Will use CSV data instead
elif task == "Comparative Temperature Estimation":
    task_type = "comparative_temp_estimation"
    task_dirs = None  # Will use CSV data instead

# Initialize task progress if not exists
if task not in st.session_state.task_progress:
    if task_type in ["thermal_reasoning_1", "thermal_reasoning_2"]:
        # Load CSV data for thermal reasoning tasks
        csv_file = "double_sampled.csv" if task_type == "thermal_reasoning_1" else "single_sampled.csv"
        current_data = get_thermal_reasoning_data(csv_file, task_type)
        if current_data:
            st.session_state.task_progress[task] = {
                'images': current_data,  # Store task data instead of image paths
                'current_index': 0,
                'completed': False,
                'task_type': task_type
            }
        else:
            st.warning(f"No data found for {task}.")
            st.stop()
    elif task_type in ["arrow_temp_estimation", "semantic_temp_estimation", "comparative_temp_estimation"]:
        # Load CSV data for temperature estimation tasks
        if task_type == "arrow_temp_estimation":
            csv_file = "arrow_sampled.csv"
        else:
            csv_file = "region_sampled.csv"
        
        current_data = get_temperature_estimation_data(csv_file, task_type)
        if current_data:
            st.session_state.task_progress[task] = {
                'images': current_data,  # Store task data instead of image paths
                'current_index': 0,
                'completed': False,
                'task_type': task_type
            }
        else:
            st.warning(f"No data found for {task}.")
            st.stop()
    else:
        # Load image data for regular tasks
        current_images = get_all_images(task_dirs)
        if current_images:
            st.session_state.task_progress[task] = {
                'images': current_images,
                'current_index': 0,
                'completed': False,
                'task_type': task_type
            }
        else:
            st.warning(f"No images found in the {task} folder(s).")
            st.stop()

# Get current task progress
task_data = st.session_state.task_progress[task]
task_images = task_data['images']
current_index = task_data['current_index']
task_completed = task_data['completed']
task_type = task_data['task_type']

# Update current task
st.session_state.current_task = task

# Display current progress
total_images = len(task_images)

if task_completed:
    st.success(f"{task} completed! You've evaluated all {total_images} images.")
    
    # Suggest next task
    all_tasks = [
        "Modality identification", 
        "Human Counting", 
        "Thermal Reasoning 1", 
        "Thermal Reasoning 2",
        "Arrow based temp estimation",
        "Semantic Temperature Estimation",
        "Comparative Temperature Estimation"
    ]
    
    # Find incomplete tasks
    incomplete_tasks = []
    for task_name in all_tasks:
        if task_name in st.session_state.task_progress:
            if not st.session_state.task_progress[task_name]['completed']:
                incomplete_tasks.append(task_name)
        else:
            incomplete_tasks.append(task_name)
    
    if incomplete_tasks:
        st.info(f"🎯 **Next up:** Try '{incomplete_tasks[0]}' from the sidebar!")
        st.markdown("**Remaining tasks:**")
        for i, incomplete_task in enumerate(incomplete_tasks, 1):
            st.markdown(f"{i}. {incomplete_task}")
    else:
        st.balloons()
        st.success("🎉 **Congratulations! You've completed all tasks!**")
        st.markdown("**Thank you for your participation in this human baseline evaluation.**")
    
    if st.button("Restart this task"):
        # Reset this specific task
        st.session_state.task_progress[task]['current_index'] = 0
        st.session_state.task_progress[task]['completed'] = False
        # Re-shuffle images for restart
        if task == "Modality identification":
            current_images = get_all_images(["data/task1", "data/task2"])
            st.session_state.task_progress[task]['images'] = current_images
        elif task == "Human Counting":
            current_images = get_all_images(["data/task3"])
            st.session_state.task_progress[task]['images'] = current_images
        elif task == "Thermal Reasoning 1":
            current_data = get_thermal_reasoning_data("double_sampled.csv", "thermal_reasoning_1")
            st.session_state.task_progress[task]['images'] = current_data
        elif task == "Thermal Reasoning 2":
            current_data = get_thermal_reasoning_data("single_sampled.csv", "thermal_reasoning_2")
            st.session_state.task_progress[task]['images'] = current_data
        elif task == "Arrow based temp estimation":
            current_data = get_temperature_estimation_data("arrow_sampled.csv", "arrow_temp_estimation")
            st.session_state.task_progress[task]['images'] = current_data
        elif task == "Semantic Temperature Estimation":
            current_data = get_temperature_estimation_data("region_sampled.csv", "semantic_temp_estimation")
            st.session_state.task_progress[task]['images'] = current_data
        elif task == "Comparative Temperature Estimation":
            current_data = get_temperature_estimation_data("region_sampled.csv", "comparative_temp_estimation")
            st.session_state.task_progress[task]['images'] = current_data
        st.rerun()
    st.stop()

st.sidebar.write(f"Progress: {current_index + 1}/{total_images}")
st.sidebar.progress((current_index + 1) / total_images)

# Add overall progress summary in sidebar
st.sidebar.markdown("---")
st.sidebar.write("**Overall Progress:**")
for task_name, progress in st.session_state.task_progress.items():
    total = len(progress['images'])
    current = progress['current_index'] + (1 if progress['completed'] else 0)
    status = "[DONE]" if progress['completed'] else "[IN PROGRESS]"
    st.sidebar.write(f"{status} {task_name}: {min(current, total)}/{total}")

# Display current image and collect answer
if task_type == "thermal_classification":
    st.header(f"{task}: Is it a thermal image?")
    
    # For merged tasks, images are stored as (full_path, filename) tuples
    if isinstance(task_images[current_index], tuple):
        image_path, current_image_name = task_images[current_index]
    else:
        current_image_name = task_images[current_index]
        image_path = current_image_name
    
    # Start timing when image is displayed (only once per image)
    timer_key = f"{task}_{current_index}"
    if st.session_state.image_start_time != timer_key:
        st.session_state.image_start_time = timer_key
        st.session_state[f"start_time_{timer_key}"] = time.time()
    
    # Create two columns: left for image, right for buttons
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Display image
        image = Image.open(image_path)
        st.image(image, caption=f"Image {current_index + 1}/{total_images}", use_container_width=True)
    
    with col2:
        st.write("Your Answer:")
        
        # Thermal button
        if st.button("Thermal", key=f"thermal_{task}_{current_index}", use_container_width=True):
            # Calculate response time
            end_time = time.time()
            start_time = st.session_state[f"start_time_{timer_key}"]
            response_time = round((end_time - start_time) * 1000, 1)  # Convert to milliseconds
            
            # Save result
            save_results(
                user_id=st.session_state.user_id,
                user_info=st.session_state.user_info,
                task=task,
                image_name=current_image_name,
                answer="Thermal",
                response_time=response_time
            )
            
            st.success(f"Answer saved: Thermal (Time: {response_time}ms)")
            
            # Clean up timing data
            del st.session_state[f"start_time_{timer_key}"]
            st.session_state.image_start_time = None
            
            # Move to next image
            if current_index + 1 < total_images:
                st.session_state.task_progress[task]['current_index'] += 1
                st.rerun()
            else:
                st.session_state.task_progress[task]['completed'] = True
                st.rerun()
        
        # Not Thermal button
        if st.button("Not Thermal", key=f"not_thermal_{task}_{current_index}", use_container_width=True):
            # Calculate response time
            end_time = time.time()
            start_time = st.session_state[f"start_time_{timer_key}"]
            response_time = round((end_time - start_time) * 1000, 1)  # Convert to milliseconds
            
            # Save result
            save_results(
                user_id=st.session_state.user_id,
                user_info=st.session_state.user_info,
                task=task,
                image_name=current_image_name,
                answer="Not Thermal",
                response_time=response_time
            )
            
            st.success(f"Answer saved: Not Thermal (Time: {response_time}ms)")
            
            # Clean up timing data
            del st.session_state[f"start_time_{timer_key}"]
            st.session_state.image_start_time = None
            
            # Move to next image
            if current_index + 1 < total_images:
                st.session_state.task_progress[task]['current_index'] += 1
                st.rerun()
            else:
                st.session_state.task_progress[task]['completed'] = True
                st.rerun()

elif task_type == "people_counting":
    st.header("Human Counting: Count the number of people")
    
    # For single task, images are stored as (full_path, filename) tuples
    if isinstance(task_images[current_index], tuple):
        image_path, current_image_name = task_images[current_index]
    else:
        current_image_name = task_images[current_index]
        image_path = current_image_name
    
    # Start timing when image is displayed (only once per image)
    timer_key = f"{task}_{current_index}"
    if st.session_state.image_start_time != timer_key:
        st.session_state.image_start_time = timer_key
        st.session_state[f"start_time_{timer_key}"] = time.time()
    
    # Create two columns: left for image, right for input
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Display image
        image = Image.open(image_path)
        st.image(image, caption=f"Image {current_index + 1}/{total_images}", use_container_width=True)
    
    with col2:
        st.write("Your Answer:")
        
        # Answer collection
        count = st.number_input("Number of people:", min_value=0, step=1, key=f"count_{task}_{current_index}")
        
        if st.button("Submit Count", key=f"submit_count_{task}_{current_index}", use_container_width=True):
            # Calculate response time
            end_time = time.time()
            start_time = st.session_state[f"start_time_{timer_key}"]
            response_time = round((end_time - start_time) * 1000, 1)  # Convert to milliseconds
            
            # Save result
            save_results(
                user_id=st.session_state.user_id,
                user_info=st.session_state.user_info,
                task=task,
                image_name=current_image_name,
                answer=str(count),
                response_time=response_time
            )
            
            st.success(f"Count saved: {count} (Time: {response_time}ms)")
            
            # Clean up timing data
            del st.session_state[f"start_time_{timer_key}"]
            st.session_state.image_start_time = None
            
            # Move to next image
            if current_index + 1 < total_images:
                st.session_state.task_progress[task]['current_index'] += 1
                st.rerun()
            else:
                st.session_state.task_progress[task]['completed'] = True
                st.rerun()

elif task_type == "thermal_reasoning_1":
    st.header("Thermal Reasoning 1: Temperature Comparison")
    
    # Get current task data
    current_data = task_images[current_index]
    image_path = current_data['image_path']
    current_image_name = current_data['image_name']
    body_part = current_data['body_part']
    question = current_data['question']
    
    # Display question above the image
    st.markdown(f"### Question: {question}")
    
    # Start timing when image is displayed (only once per image)
    timer_key = f"{task}_{current_index}"
    if st.session_state.image_start_time != timer_key:
        st.session_state.image_start_time = timer_key
        st.session_state[f"start_time_{timer_key}"] = time.time()
    
    # Create two columns: left for image, right for buttons
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Display image
        image = Image.open(image_path)
        st.image(image, caption=f"Image {current_index + 1}/{total_images}", use_container_width=True)
    
    with col2:
        st.write("Your Answer:")
        
        # Left person button
        if st.button("Left Person", key=f"left_{task}_{current_index}", use_container_width=True):
            # Calculate response time
            end_time = time.time()
            start_time = st.session_state[f"start_time_{timer_key}"]
            response_time = round((end_time - start_time) * 1000, 1)  # Convert to milliseconds
            
            # Save result
            save_results(
                user_id=st.session_state.user_id,
                user_info=st.session_state.user_info,
                task=task,
                image_name=current_image_name,
                answer="Left",
                response_time=response_time
            )
            
            st.success(f"Answer saved: Left Person (Time: {response_time}ms)")
            
            # Clean up timing data
            del st.session_state[f"start_time_{timer_key}"]
            st.session_state.image_start_time = None
            
            # Move to next image
            if current_index + 1 < total_images:
                st.session_state.task_progress[task]['current_index'] += 1
                st.rerun()
            else:
                st.session_state.task_progress[task]['completed'] = True
                st.rerun()
        
        # Right person button
        if st.button("Right Person", key=f"right_{task}_{current_index}", use_container_width=True):
            # Calculate response time
            end_time = time.time()
            start_time = st.session_state[f"start_time_{timer_key}"]
            response_time = round((end_time - start_time) * 1000, 1)  # Convert to milliseconds
            
            # Save result
            save_results(
                user_id=st.session_state.user_id,
                user_info=st.session_state.user_info,
                task=task,
                image_name=current_image_name,
                answer="Right",
                response_time=response_time
            )
            
            st.success(f"Answer saved: Right Person (Time: {response_time}ms)")
            
            # Clean up timing data
            del st.session_state[f"start_time_{timer_key}"]
            st.session_state.image_start_time = None
            
            # Move to next image
            if current_index + 1 < total_images:
                st.session_state.task_progress[task]['current_index'] += 1
                st.rerun()
            else:
                st.session_state.task_progress[task]['completed'] = True
                st.rerun()

elif task_type == "thermal_reasoning_2":
    st.header("Thermal Reasoning 2: Temperature Ordering")
    
    # Get current task data
    current_data = task_images[current_index]
    image_path = current_data['image_path']
    current_image_name = current_data['image_name']
    correct_order = current_data['correct_order']
    
    # Display task instructions above the image
    # Sort the body parts to ensure consistent display order
    sorted_parts = sorted(correct_order)
    st.markdown(f"### Task: Arrange these body parts from **hottest to coldest**:")
    st.markdown(f"**Body parts:** {', '.join(sorted_parts)}")
    
    # Start timing when image is displayed (only once per image)
    timer_key = f"{task}_{current_index}"
    if st.session_state.image_start_time != timer_key:
        st.session_state.image_start_time = timer_key
        st.session_state[f"start_time_{timer_key}"] = time.time()
    
    # Create two columns: left for image, right for input
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Display image
        image = Image.open(image_path)
        st.image(image, caption=f"Image {current_index + 1}/{total_images}", use_container_width=True)
    
    with col2:
        st.write("Your Answer:")
        st.write("Drag to reorder:")
        
        # Create a simple ordering interface using selectboxes
        ordered_parts = []
        available_parts = correct_order.copy()
        
        for i in range(len(correct_order)):
            if available_parts:
                selected = st.selectbox(
                    f"Position {i+1} (hottest to coldest):",
                    options=available_parts,
                    key=f"order_{task}_{current_index}_{i}"
                )
                ordered_parts.append(selected)
                if selected in available_parts:
                    available_parts.remove(selected)
        
        if st.button("Submit Order", key=f"submit_order_{task}_{current_index}", use_container_width=True):
            # Calculate response time
            end_time = time.time()
            start_time = st.session_state[f"start_time_{timer_key}"]
            response_time = round((end_time - start_time) * 1000, 1)  # Convert to milliseconds
            
            # Save result as comma-separated string
            answer = ",".join(ordered_parts)
            
            save_results(
                user_id=st.session_state.user_id,
                user_info=st.session_state.user_info,
                task=task,
                image_name=current_image_name,
                answer=answer,
                response_time=response_time
            )
            
            st.success(f"Order saved: {' → '.join(ordered_parts)} (Time: {response_time}ms)")
            
            # Clean up timing data
            del st.session_state[f"start_time_{timer_key}"]
            st.session_state.image_start_time = None
            
            # Move to next image
            if current_index + 1 < total_images:
                st.session_state.task_progress[task]['current_index'] += 1
                st.rerun()
            else:
                st.session_state.task_progress[task]['completed'] = True
                st.rerun()

elif task_type == "arrow_temp_estimation":
    st.header("Arrow Based Temperature Estimation")
    
    # Get current task data
    current_data = task_images[current_index]
    image_path = current_data['image_path']
    current_image_name = current_data['image_name']
    x_coord = current_data['x']
    y_coord = current_data['y']
    actual_temp = current_data['actual_temperature']
    question = current_data['question']
    
    # Display question above the image
    st.markdown(f"### {question}")
    st.markdown("**Use the red arrow as a guide to estimate the temperature at that specific location.**")
    st.markdown("**Tip:** Use the temperature scale/colormap shown in the image to help estimate the temperature.")
    
    # Start timing when image is displayed (only once per image)
    timer_key = f"{task}_{current_index}"
    if st.session_state.image_start_time != timer_key:
        st.session_state.image_start_time = timer_key
        st.session_state[f"start_time_{timer_key}"] = time.time()
    
    # Create two columns: left for image, right for input
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Draw arrow on image and display
        image_with_arrow = draw_arrow_on_image(image_path, x_coord, y_coord)
        st.image(image_with_arrow, caption=f"Image {current_index + 1}/{total_images}", use_container_width=True)
        st.caption(f"Arrow points to coordinates: ({x_coord}, {y_coord})")
    
    with col2:
        st.write("Your Answer:")
        
        # Temperature estimation input
        estimated_temp = st.number_input(
            "Estimated Temperature (°C):", 
            min_value=0.0, 
            max_value=50.0, 
            step=0.1, 
            format="%.1f",
            key=f"temp_{task}_{current_index}"
        )
        
        if st.button("Submit Estimate", key=f"submit_temp_{task}_{current_index}", use_container_width=True):
            # Calculate response time
            end_time = time.time()
            start_time = st.session_state[f"start_time_{timer_key}"]
            response_time = round((end_time - start_time) * 1000, 1)  # Convert to milliseconds
            
            # Save result
            save_results(
                user_id=st.session_state.user_id,
                user_info=st.session_state.user_info,
                task=task,
                image_name=current_image_name,
                answer=str(estimated_temp),
                response_time=response_time
            )
            
            # Calculate error for feedback
            error = abs(estimated_temp - actual_temp)
            st.success(f"Estimate saved: {estimated_temp}°C (Time: {response_time}ms)")
            st.info(f"Actual temperature: {actual_temp}°C | Error: {error:.1f}°C")
            
            # Clean up timing data
            del st.session_state[f"start_time_{timer_key}"]
            st.session_state.image_start_time = None
            
            # Move to next image
            if current_index + 1 < total_images:
                st.session_state.task_progress[task]['current_index'] += 1
                st.rerun()
            else:
                st.session_state.task_progress[task]['completed'] = True
                st.rerun()

elif task_type == "semantic_temp_estimation":
    st.header("Semantic Temperature Estimation")
    
    # Get current task data
    current_data = task_images[current_index]
    image_path = current_data['image_path']
    current_image_name = current_data['image_name']
    body_part = current_data['body_part']
    actual_temp = current_data['actual_temperature']
    question = current_data['question']
    
    # Display question above the image
    st.markdown(f"### {question}")
    st.markdown(f"**Focus on the {body_part} in the image and estimate its temperature.**")
    st.markdown("**Tip:** Use the temperature scale/colormap shown in the image to help estimate the temperature.")
    
    # Start timing when image is displayed (only once per image)
    timer_key = f"{task}_{current_index}"
    if st.session_state.image_start_time != timer_key:
        st.session_state.image_start_time = timer_key
        st.session_state[f"start_time_{timer_key}"] = time.time()
    
    # Create two columns: left for image, right for input
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Display image
        image = Image.open(image_path)
        st.image(image, caption=f"Image {current_index + 1}/{total_images}", use_container_width=True)
    
    with col2:
        st.write("Your Answer:")
        
        # Temperature estimation input
        estimated_temp = st.number_input(
            f"Temperature of {body_part} (°C):", 
            min_value=0.0, 
            max_value=50.0, 
            step=0.1, 
            format="%.1f",
            key=f"semantic_temp_{task}_{current_index}"
        )
        
        if st.button("Submit Estimate", key=f"submit_semantic_{task}_{current_index}", use_container_width=True):
            # Calculate response time
            end_time = time.time()
            start_time = st.session_state[f"start_time_{timer_key}"]
            response_time = round((end_time - start_time) * 1000, 1)  # Convert to milliseconds
            
            # Save result
            save_results(
                user_id=st.session_state.user_id,
                user_info=st.session_state.user_info,
                task=task,
                image_name=current_image_name,
                answer=str(estimated_temp),
                response_time=response_time
            )
            
            # Calculate error for feedback
            error = abs(estimated_temp - actual_temp)
            st.success(f"Estimate saved: {estimated_temp}°C (Time: {response_time}ms)")
            st.info(f"Actual temperature: {actual_temp:.1f}°C | Error: {error:.1f}°C")
            
            # Clean up timing data
            del st.session_state[f"start_time_{timer_key}"]
            st.session_state.image_start_time = None
            
            # Move to next image
            if current_index + 1 < total_images:
                st.session_state.task_progress[task]['current_index'] += 1
                st.rerun()
            else:
                st.session_state.task_progress[task]['completed'] = True
                st.rerun()

elif task_type == "comparative_temp_estimation":
    st.header("Comparative Temperature Estimation")
    
    # Get current task data
    current_data = task_images[current_index]
    image_path = current_data['image_path']
    current_image_name = current_data['image_name']
    body_part = current_data['body_part']
    side = current_data['side']
    actual_temp = current_data['actual_temperature']
    question = current_data['question']
    
    # Display question above the image
    st.markdown(f"### {question}")
    st.markdown(f"**Focus on the {side} person's {body_part} and estimate its temperature.**")
    st.markdown("**Tip:** Use the temperature scale/colormap shown in the image to help estimate the temperature.")
    
    # Start timing when image is displayed (only once per image)
    timer_key = f"{task}_{current_index}"
    if st.session_state.image_start_time != timer_key:
        st.session_state.image_start_time = timer_key
        st.session_state[f"start_time_{timer_key}"] = time.time()
    
    # Create two columns: left for image, right for input
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Display image
        image = Image.open(image_path)
        st.image(image, caption=f"Image {current_index + 1}/{total_images}", use_container_width=True)
    
    with col2:
        st.write("Your Answer:")
        
        # Temperature estimation input
        estimated_temp = st.number_input(
            f"{side.title()} person's {body_part} temp (°C):", 
            min_value=0.0, 
            max_value=50.0, 
            step=0.1, 
            format="%.1f",
            key=f"comparative_temp_{task}_{current_index}"
        )
        
        if st.button("Submit Estimate", key=f"submit_comparative_{task}_{current_index}", use_container_width=True):
            # Calculate response time
            end_time = time.time()
            start_time = st.session_state[f"start_time_{timer_key}"]
            response_time = round((end_time - start_time) * 1000, 1)  # Convert to milliseconds
            
            # Save result
            save_results(
                user_id=st.session_state.user_id,
                user_info=st.session_state.user_info,
                task=task,
                image_name=current_image_name,
                answer=str(estimated_temp),
                response_time=response_time
            )
            
            # Calculate error for feedback
            error = abs(estimated_temp - actual_temp)
            st.success(f"Estimate saved: {estimated_temp}°C (Time: {response_time}ms)")
            st.info(f"Actual temperature: {actual_temp:.1f}°C | Error: {error:.1f}°C")
            
            # Clean up timing data
            del st.session_state[f"start_time_{timer_key}"]
            st.session_state.image_start_time = None
            
            # Move to next image
            if current_index + 1 < total_images:
                st.session_state.task_progress[task]['current_index'] += 1
                st.rerun()
            else:
                st.session_state.task_progress[task]['completed'] = True
                st.rerun()
