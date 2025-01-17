import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import streamlit as st

# Define a helper function for execution
def execute_function(func, args):
    return func(*args)

def run_concurrently(funcs_with_args):
    """
    Run a list of functions with their respective arguments concurrently, using at most (n-2) cores.

    Args:
        funcs_with_args (list): A list of tuples. Each tuple contains:
            - A callable function
            - A tuple of positional arguments for the function

    Returns:
        list: The results of all the functions in the order they were given.
    """
    # Calculate the number of available cores
    num_cores = multiprocessing.cpu_count()
    max_workers = max(num_cores - 2, 1)  # Ensure at least one worker is used

    progress_bar = st.progress(0, text=f"CPU Cores: `{max_workers}`\n Progress:")

    results = []

    i = 0
    len_funcs = len(funcs_with_args)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit tasks to the pool
        future_to_func = {
            executor.submit(partial(execute_function, func, args)): (func, args)
            for func, args in funcs_with_args
        }

        # Collect results as they complete
        for future in as_completed(future_to_func):
            i += 1
            progress_bar.progress(value=((i)/len_funcs), text=f"CPU Cores: `{max_workers}`\n Progress: {(100*(i)/len_funcs):.1f}%")
            results.append(future.result())
    progress_bar.empty()
    return results