import board
import busio
import adafruit_mlx90640
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Initialize I2C bus
i2c = busio.I2C(board.SCL, board.SDA, frequency=400000) # Use the baudrate you set

# Initialize MLX90640 camera
mlx = adafruit_mlx90640.MLX90640(i2c)
print("MLX90640 detected, serial number:", [hex(i) for i in mlx.serial_number])

# Set refresh rate (e.g., 16 Hz, options are 0.5, 1, 2, 4, 8, 16, 32, 64)
# Higher refresh rates require higher I2C baudrate and more processing power
mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_16_HZ

# Create a frame buffer to store thermal data
frame = np.zeros((24 * 32,)) # MLX90640 is 32x24 pixels

# Setup the plot for live visualization
fig, ax = plt.subplots(figsize=(6, 8)) # Adjust figure size as needed for 24x32
im = ax.imshow(np.zeros((24, 32)), cmap='inferno', vmin=20, vmax=35) # Initial empty image
plt.colorbar(im, ax=ax, label='Temperature (°C)')
ax.set_title('MLX90640 Thermal Stream')
ax.set_aspect('equal')

def update(frame_num):
    try:
        mlx.getFrame(frame)
        # Reshape the 1D array into a 24x32 2D array (height x width)
        # The MLX90640 reads data in a specific order that might require
        # transposing or reordering depending on your desired orientation.
        # For Adafruit's library, direct reshape usually works for the default orientation.
        temp_data = np.reshape(frame, (24, 32))

        # You might need to rotate or flip the image depending on your camera's orientation
        # For example:
        # temp_data = np.rot90(temp_data, k=1) # Rotate 90 degrees clockwise
        # temp_data = np.flipud(temp_data) # Flip vertically

        im.set_array(temp_data)
        # Optionally update colorbar limits dynamically based on min/max temperatures
        # im.set_clim(np.min(temp_data) - 1, np.max(temp_data) + 1)
        return im,
    except ValueError:
        # These happen if the I2C bus is busy, just retry
        return im,

ani = FuncAnimation(fig, update, interval=50, blit=True) # interval in milliseconds (e.g., 50ms for 20 FPS)
plt.show()
