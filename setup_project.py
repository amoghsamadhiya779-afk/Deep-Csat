import os
import shutil
import logging
import subprocess
import sys
import platform

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

def create_project_structure():
    """
    Creates a standard data science project structure and moves files 
    to their appropriate locations.
    """
    logging.info("--- 1. Organizing Project Structure ---")
    
    # 1. Define the directory structure
    directories = [
        "data",          # For raw and processed data
        "src",           # For source code
        "models",        # For saved models (.pkl)
        "notebooks",     # For Jupyter notebooks
        "logs"           # For log files
    ]

    # 2. Create Directories
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logging.info(f"✔ Verified/Created directory: {directory}/")

    # 3. Define file moves (Source -> Destination Folder)
    files_to_move = {
        "eCommerce_Customer_support_data.csv": "data",
        "Sample_ML_Submission_Template-2.ipynb": "notebooks",
        "DeepCSAT – Ecommerce.pptx": "notebooks",
        "csat_model.pkl": "models" 
    }

    # 4. Move Files
    for filename, target_folder in files_to_move.items():
        if os.path.exists(filename):
            destination = os.path.join(target_folder, filename)
            if not os.path.exists(destination):
                shutil.move(filename, destination)
                logging.info(f"➜ Moved '{filename}' to '{target_folder}/'")
            else:
                logging.info(f"⚠ '{filename}' already exists in '{target_folder}/'. Skipping move.")

    # 5. Create __init__.py in src
    init_path = os.path.join("src", "__init__.py")
    if not os.path.exists(init_path):
        with open(init_path, "w") as f:
            pass
        logging.info("✔ Created 'src/__init__.py'")

def setup_virtual_environment():
    """
    Creates a virtual environment and installs requirements.
    """
    logging.info("\n--- 2. Setting up Virtual Environment ---")
    
    venv_name = ".venv"
    
    # 1. Create venv if it doesn't exist
    if not os.path.exists(venv_name):
        logging.info(f"Creating virtual environment '{venv_name}'...")
        try:
            subprocess.check_call([sys.executable, "-m", "venv", venv_name])
            logging.info("✔ Virtual environment created.")
        except subprocess.CalledProcessError:
            logging.error("❌ Failed to create virtual environment.")
            return
    else:
        logging.info(f"✔ Virtual environment '{venv_name}' already exists.")

    # 2. Determine paths for pip and python executables inside venv
    if platform.system() == "Windows":
        pip_executable = os.path.join(venv_name, "Scripts", "pip")
        python_executable = os.path.join(venv_name, "Scripts", "python")
        activation_cmd = f"{venv_name}\\Scripts\\activate"
    else: # Linux/Mac
        pip_executable = os.path.join(venv_name, "bin", "pip")
        python_executable = os.path.join(venv_name, "bin", "python")
        activation_cmd = f"source {venv_name}/bin/activate"

    # 3. Install Requirements
    req_file = "requirements.txt"
    if os.path.exists(req_file):
        logging.info(f"Installing dependencies from {req_file}...")
        try:
            # Upgrade pip first
            subprocess.check_call([python_executable, "-m", "pip", "install", "--upgrade", "pip"])
            # Install requirements
            subprocess.check_call([pip_executable, "install", "-r", req_file])
            logging.info("✔ Dependencies installed successfully.")
        except subprocess.CalledProcessError:
            logging.error("❌ Failed to install dependencies.")
    else:
        logging.warning(f"⚠ '{req_file}' not found. Skipping dependency installation.")

    logging.info("\n--- Setup Complete ---")
    logging.info(f"To activate the environment, run: {activation_cmd}")
    logging.info("Then run the pipeline with: python main.py")

if __name__ == "__main__":
    create_project_structure()
    setup_virtual_environment()