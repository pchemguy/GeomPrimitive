import os
import random
import time
from typing import Tuple, Union  # Added for type hints

from multiprocessing import Pool

import numpy as np

import matplotlib
# Use a non-interactive backend (safer/faster in workers). Set this before importing pyplot
# must be before importing pyplot OR set env var MPLBACKEND=Agg
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from scipy.interpolate import splprep, splev

# Type alias for file paths
PathLike = Union[str, os.PathLike]


def _seed_worker():
    """
    Per-process randomness to avoid identical draws, seed in each worker via an initializer
    """
    random.seed(os.getpid() ^ int(time.time()))
    np.random.seed((os.getpid() * 2654435761) % 2**32)

    
def generate_synthetic_image(output_filepath: PathLike, img_size: Tuple[int, int] = (1920, 1080)) -> PathLike:
    """
    Generates a single synthetic, 'hand-drawn' image using Matplotlib
    and saves it to a file.
    """
    
    # 1. Calculate DPI for the correct pixel output
    # Matplotlib works in inches, so we need to convert pixels
    dpi: int = 100 
    fig_width: float = img_size[0] / dpi
    fig_height: float = img_size[1] / dpi

    # 2. Use the 'xkcd' style context manager
    # This is the magic! All lines drawn inside this block
    # will be 'imperfect'.
    with plt.xkcd():
        
        # 3. Create the figure and axes
        # We make it 'frameon=False' to remove borders
        fig: Figure
        ax: Axes
        fig, ax = plt.subplots(figsize=(fig_width, fig_height), frameon=False)
        
        # 4. Configure the canvas
        # Set limits to match pixel dimensions
        ax.set_xlim(0, img_size[0])
        ax.set_ylim(0, img_size[1])
        # Invert Y-axis to match image (0,0 at top-left)
        ax.invert_yaxis() 
        # Turn off the axis lines and labels completely
        ax.axis('off') 

        # 5. Draw your shapes!
        # Matplotlib's high-level functions make this easy.
        
        # Example 1: A "hand-drawn" wobbly spline
        pts: np.ndarray = np.array([
            [random.randint(0, 400), random.randint(0, 400)],
            [random.randint(400, 800), random.randint(500, 900)],
            [random.randint(900, 1800), random.randint(100, 500)]
        ])
        # Ensure m > k holds: number of points > spline degree
        m = len(pts)
        # Degree must be < number of points
        k = min(3, len(pts) - 1)        
        
        x_new: np.ndarray
        y_new: np.ndarray
        if m > k:
            # tck: tuple (knots, coefficients, degree)
            # u: parameter values
            tck: tuple
            u: np.ndarray
            tck, u = splprep([pts[:, 0], pts[:, 1]], s=0, k=k)
            x_new, y_new = splev(np.linspace(0, 1, 100), tck)
        else:
            # fallback to straight line if not enough points
            x_new, y_new = (pts[:, 0], pts[:, 1])

        ax.plot(x_new, y_new, 'b-') # 'b-' is a blue line

        # Example 2: A "hand-drawn" imperfect oval
        oval: Ellipse = Ellipse(
            xy=(random.randint(800, 1200), random.randint(600, 900)), 
            width=random.randint(100, 500), 
            height=random.randint(100, 500),
            angle=random.uniform(0, 90),
            edgecolor='r', 
            facecolor='none'
        )
        ax.add_patch(oval)

        # Example 3: A "hand-drawn" dashed line
        ax.plot(
            [10, 1800],                              # x1, x2
            [random.randint(10, 1000), 1000],       # y1, y2
            'g--',                                   # 'g--' is a green dashed line
            linewidth=random.uniform(1, 3)
        )

        # 6. Save the final image
        # 'bbox_inches' and 'pad_inches' are critical to remove
        # all whitespace and padding around the image.
        fig.savefig(
            output_filepath, 
            dpi=dpi,
            format='jpg',
            bbox_inches='tight', 
            pad_inches=0
        )
        
        # 7. CRITICAL: Close the figure to free memory
        plt.close(fig)
        
    return output_filepath


def main():
    # --- Configuration ---
    BATCH_SIZE = 12
    OUTPUT_DIR = "./synth_images_prototype"
    IMG_HEIGHT = 1080
    IMG_WIDTH = 1920
    
    # Use 75% of CPU cores, as you specified
    num_cores = max(1, int(os.cpu_count() * 0.75))
    print(f"Starting prototype generation...")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Output Dir: {OUTPUT_DIR}")
    print(f"Using {num_cores} parallel processes.")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # --- Prepare the list of "jobs" ---
    # We create a list of all the output filepaths we want to generate.
    # The pool will map our function to this list.
    job_filepaths = []
    for i in range(BATCH_SIZE):
        f_name = f"prototype_image_{i:06d}.jpg"
        job_filepaths.append(os.path.join(OUTPUT_DIR, f_name))

    # --- Run the Parallel Pool ---
    start_time = time.time()
    
    # Create the pool of worker processes
    with Pool(processes=num_cores, initializer=_seed_worker) as pool:
        # pool.map will take the list 'job_filepaths' and
        # automatically feed one to each call of our function,
        # in parallel.
        # We use 'chunksize=10' to feed jobs to workers in
        # chunks, which is more efficient than one-by-one.
        print("Submitting jobs to pool...")
        
        # We need a way to pass the static img_size.
        # We can use functools.partial or a simple lambda.
        from functools import partial
        
        generation_task = partial(
            generate_synthetic_image, 
            img_size=(IMG_WIDTH, IMG_HEIGHT)
        )
        
        # This is the main work loop
        results = pool.map(generation_task, job_filepaths, chunksize=10)
    
    end_time = time.time()
    duration = end_time - start_time
    imgs_per_sec = BATCH_SIZE / duration
    
    print("\n--- Generation Complete ---")
    print(f"Generated {BATCH_SIZE} images in {duration:.2f} seconds.")
    print(f"Throughput: {imgs_per_sec:.2f} images/sec")


if __name__ == "__main__":
    # This 'if' block is essential for multiprocessing
    main()
