#!/usr/bin/env python
import sys
import warnings
from datetime import datetime
from stockmarketagent2.crew import Stockmarketagent2

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# This main file is intended to be a way for you to run your
# crew locally, so refrain from adding unnecessary logic into this file.
# Replace with inputs you want to test with, it will automatically
# interpolate any tasks and agents information

def run():
    """
    Run the crew with user input for ticker and years.
    """
    ticker = input("📈 Enter stock ticker (e.g. AAPL): ").upper()
    years = float(input("📅 Enter number of years to analyze (e.g. 1, 0.5, 3): "))

    inputs = {
        'ticker': ticker,
        'years': years
    }

    try:
        Stockmarketagent2().crew().kickoff(inputs=inputs)
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")

def train():
    """
    Train the crew for a given number of iterations.
    """
    inputs = {
        "ticker": sys.argv[3].upper(),
        "years": float(sys.argv[4])
    }

    try:
        Stockmarketagent2().crew().train(n_iterations=int(sys.argv[1]), filename=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")

def replay():
    """
    Replay the crew execution from a specific task.
    """
    try:
        Stockmarketagent2().crew().replay(task_id=sys.argv[1])

    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")

def test():
    ticker = input("📈 Enter stock ticker (e.g. AAPL): ").upper()
    years = float(input("📅 Enter number of years to analyze (e.g. 1, 0.5, 3): "))

    inputs = {
        'ticker': ticker,
        'years': years
    }

    try:
        Stockmarketagent2().crew().kickoff(inputs=inputs)
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")
