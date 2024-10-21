#!/bin/bash

# Create a virtual environment
python3.12 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Install the required packages
pip3.12 install -r requirements.txt


# Run python program
python app/main.py

# langgraph studio run
# upload the project in langgraph studio and open doker
