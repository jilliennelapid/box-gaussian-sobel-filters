import customtkinter as ctk
from tkinter import filedialog, messagebox
import cv2
from PIL import Image
import numpy as np
import math

class FilterApp(ctk.CTkFrame):
    def __init__(self, parent):
        super().__init__(parent)

        self.window_size = ctk.StringVar()

        parent.title("Homework 2: Applying Filters")
        parent.geometry("800x600")
        parent.rowconfigure(0, weight=1)
        parent.columnconfigure(0, weight=1)

        self.configure(fg_color="transparent")
        self.grid(row=0, column=0, sticky="nsew")
        self.rowconfigure(0, weight=0)
        self.rowconfigure(1, weight=0)
        self.rowconfigure(2, weight=1)
        self.columnconfigure(0, weight=1)

        self.top_frame = ctk.CTkFrame(self)
        self.top_frame.grid(row=0, column=0, pady=(20,10), sticky="nsew")
        self.top_frame.rowconfigure(0, weight=1)
        self.top_frame.columnconfigure(0, weight=1)
        self.top_frame.columnconfigure(1, weight=1)

        self.load_button = ctk.CTkButton(self.top_frame, corner_radius=10, text="Load Image", font=("Arial", 15, "bold"),
                                         command=self.load_image, fg_color="#5989d7", width=120)
        self.load_button.grid(row=0, column=0, rowspan=2)

        self.window_size_label = ctk.CTkLabel(self.top_frame, text="Window (Kernel) Size", font=("Arial", 14, "bold"))
        self.window_size_label.grid(row=0, column=1)

        dim_sizes = ["3x3", "5x5"]
        self.window_size.set("3x3")
        self.window_select = ctk.CTkOptionMenu(self.top_frame, values=dim_sizes, variable=self.window_size)
        self.window_select.grid(row=1, column=1)

        self.middle_frame = ctk.CTkFrame(self)
        self.middle_frame.grid(row=1, column=0, pady=10, sticky="nsew")
        self.middle_frame.rowconfigure(0, weight=1)
        self.middle_frame.columnconfigure(0, weight=1)
        self.middle_frame.columnconfigure(1, weight=1)
        self.middle_frame.columnconfigure(2, weight=1)
        self.middle_frame.columnconfigure(3, weight=1)

        self.box_filter_label = ctk.CTkLabel(self.middle_frame, text="Box Filter", font=("Arial", 14, "bold"))
        self.box_filter_label.grid(row=0, column=0, rowspan=2, sticky="ew")

        self.custom_box_button = ctk.CTkButton(self.middle_frame, text="Custom Box Filter", command=self.custom_box_filter)
        self.custom_box_button.grid(row=0, column=1)

        self.opencv_box_button = ctk.CTkButton(self.middle_frame, text="OpenCV Box Filter", command=self.opencv_box_filter)
        self.opencv_box_button.grid(row=1, column=1)

        self.gaussian_filter_label = ctk.CTkLabel(self.middle_frame, text="Gaussian Filter", font=("Arial", 14, "bold"))
        self.gaussian_filter_label.grid(row=2, column=0, sticky="ew")

        self.opencv_gaussian_button = ctk.CTkButton(self.middle_frame, text="OpenCV Gaussian Filter", command=self.opencv_gaussian_filter)
        self.opencv_gaussian_button.grid(row=2, column=1)

        self.sobel_filter_label = ctk.CTkLabel(self.middle_frame, text="Sobel Filter", font=("Arial", 14, "bold"))
        self.sobel_filter_label.grid(row=0, column=2, rowspan=4, sticky="ew")

        self.custom_sobelx_button = ctk.CTkButton(self.middle_frame, text="Custom X-Axis Sobel Filter", command=self.custom_sobelx_filter)
        self.custom_sobelx_button.grid(row=0, column=3)

        self.custom_sobely_button = ctk.CTkButton(self.middle_frame, text="Custom Y-Axis Sobel Filter", command=self.custom_sobely_filter)
        self.custom_sobely_button.grid(row=1, column=3)

        self.custom_sobelxy_button = ctk.CTkButton(self.middle_frame, text="Custom X-Axis/Y-Axis Sobel Filter", command=self.custom_sobelxy_filter)
        self.custom_sobelxy_button.grid(row=2, column=3)

        self.opencv_sobelxy_button = ctk.CTkButton(self.middle_frame, text="OpenCV X-Axis/Y-Axis Sobel Filter", command=self.opencv_sobelxy_filter)
        self.opencv_sobelxy_button.grid(row=3, column=3)

        self.bottom_frame = ctk.CTkFrame(self)
        self.bottom_frame.grid(row=2, column=0, sticky="nsew")
        self.bottom_frame.rowconfigure(0, weight=1)
        self.bottom_frame.columnconfigure(0, weight=1)
        self.bottom_frame.columnconfigure(1, weight=1)

        self.original_image = None
        self.filtered_image = None

        self.original_img_label = ctk.CTkLabel(self.bottom_frame, text="Original Image")
        self.original_img_label.grid(row=0, column=0)

        self.original_img_container = ctk.CTkLabel(self.bottom_frame, text="Load an Image")
        self.original_img_container.grid(row=1, column=0)

        self.filtered_img_label = ctk.CTkLabel(self.bottom_frame, text="Filtered Image")
        self.filtered_img_label.grid(row=0, column=1)

        self.filtered_img_container = ctk.CTkLabel(self.bottom_frame, text="Then, Select a Filter Option Above")
        self.filtered_img_container.grid(row=1, column=1)

    def display_image(self, img, label: ctk.CTkLabel):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)

        pil_img = pil_img.resize((400, 300), Image.LANCZOS)

        ctk_img = ctk.CTkImage(
            light_image=pil_img,
            dark_image=pil_img,
            size=(400, 300)
        )

        label.configure(image=ctk_img, text="")
        label.image = ctk_img

    def load_image(self):
        """
        Open a file dialog to choose an image. Read it with OpenCV and display.
        Enable processing buttons after loading.
        """
        file_path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")]
        )
        if not file_path:
            return

        image = cv2.imread(file_path)
        if image is None:
            messagebox.showerror("Error", "Failed to load image.")
            return

        self.original_image = image
        self.display_image(image, self.original_img_container)

    def get_window_size(self):
        size = self.window_size.get()

        match size:
            case "3x3":
                return 3

            case "5x5":
                return 5

    def custom_box_filter(self):
        window_size = self.get_window_size()
        blurred_image = self.box_blur(self.original_image, window_size)

        self.filtered_image = blurred_image
        self.display_image(blurred_image, self.filtered_img_container)
        self.filtered_img_label.configure(text=f"Custom Box Filter - ({self.window_size.get()}) Window")

    def opencv_box_filter(self):
        window_size = self.get_window_size()
        blurred_image = cv2.boxFilter(self.original_image, -1, (window_size, window_size),  normalize=True)

        self.filtered_image = blurred_image
        self.display_image(blurred_image, self.filtered_img_container)
        self.filtered_img_label.configure(text=f"OpenCV Box Filter - ({self.window_size.get()}) Window")

    def custom_gaussian_filter(self):
        blurred_image = self.gaussian_blur(self.original_image)

        self.filtered_image = blurred_image
        self.display_image(blurred_image, self.filtered_img_container)
        self.filtered_img_label.configure(text=f"Custom Gaussian Filter - ({self.window_size.get()}) Window")

    def opencv_gaussian_filter(self):
        window_size = self.get_window_size()
        blurred_image = cv2.GaussianBlur(self.original_image, (window_size, window_size), 0)

        self.filtered_image = blurred_image
        self.display_image(blurred_image, self.filtered_img_container)
        self.filtered_img_label.configure(text=f"OpenCV Gaussian Filter - ({self.window_size.get()}) Window")

    def gaussian_kernel(self, window_size):
        size = window_size
        sigma = 1

        # Source - https://stackoverflow.com/a/47369969
        # Posted by Joe Iddon, modified by community. See post 'Timeline' for change history
        # Retrieved 2026-02-21, License - CC BY-SA 3.0
        kernel = np.fromfunction(lambda x, y: (1 / (2 * math.pi * sigma ** 2)) * math.e ** ((-1 * ((x - (size - 1) / 2) ** 2 +
                                (y - (size - 1) / 2) ** 2)) / (2 * sigma ** 2)), (size, size))
        kernel /= np.sum(kernel)

        return kernel

    def gaussian_blur(self, image):
        window_size = self.get_window_size()
        kernel = self.gaussian_kernel(window_size)

        pad = window_size // 2
        padded = np.pad(image, pad, mode='edge')

        output = np.zeros_like(image, dtype=float)

        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                region = padded[i:i + window_size, j:j + window_size]
                output[i, j] = np.sum(region * kernel)

        return output

    def get_sobel_kernel(self, window_size, axis):
        kernel = None

        if window_size == 3:
            if axis == "x":
                kernel = np.array([
                    [-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]
                ])

            elif axis == "y":
                kernel = np.array([
                    [-1, -2, -1],
                    [0, 0, 0],
                    [1, 2, 1]
                ])

        if window_size == 5:
            if axis == "x":
                kernel = np.array([
                    [-2, -1, 0, 1, 2],
                    [-2, -1, 0, 1, 2],
                    [-2, -1, 0, 1, 2],
                    [-2, -1, 0, 1, 2],
                    [-2, -1, 0, 1, 2]
                ])

            elif axis == "y":
                kernel = np.array([
                    [-2, -2, -2, -2, -2],
                    [-1, -1, -1, -1, -1],
                    [ 0,  0,  0,  0,  0],
                    [ 1,  1,  1,  1,  1],
                    [ 2,  2,  2,  2,  2]
                ])

        return kernel

    def custom_sobelx_filter(self):
        # Convert original image to grayscale
        grey_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur (float64 output)
        blurred_img = self.gaussian_blur(grey_image)

        window_size = self.get_window_size()
        # Sobel X kernel
        sobel_x = self.get_sobel_kernel(window_size, "x")

        pad = window_size // 2
        padded = np.pad(blurred_img, pad, mode='edge')

        sobel_output = np.zeros_like(blurred_img, dtype=np.float64)

        # Manual convolution
        for i in range(blurred_img.shape[0]):
            for j in range(blurred_img.shape[1]):
                region = padded[i:i + window_size, j:j + window_size]
                sobel_output[i, j] = np.sum(region * sobel_x)

        # Take absolute value
        sobel_output = np.abs(sobel_output)

        # Avoid divide-by-zero
        max_val = sobel_output.max()
        if max_val > 0:
            sobel_output = (sobel_output / max_val) * 255

        # Convert to uint8
        sobel_output = sobel_output.astype(np.uint8)

        # Convert grayscale → BGR for display_image()
        sobel_output_bgr = cv2.cvtColor(sobel_output, cv2.COLOR_GRAY2BGR)

        self.filtered_image = sobel_output_bgr
        self.display_image(sobel_output_bgr, self.filtered_img_container)

        self.filtered_img_label.configure(text=f"Sobel X-Axis Filter - ({self.window_size.get()}) Window")

    def custom_sobely_filter(self):
        # Convert original image to grayscale
        grey_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur (float64 output)
        blurred_img = self.gaussian_blur(grey_image)

        window_size = self.get_window_size()
        # Sobel Y kernel
        sobel_y = self.get_sobel_kernel(window_size, "y")

        pad = window_size // 2
        padded = np.pad(blurred_img, pad, mode='edge')

        sobel_output = np.zeros_like(blurred_img, dtype=np.float64)

        # Manual convolution
        for i in range(blurred_img.shape[0]):
            for j in range(blurred_img.shape[1]):
                region = padded[i:i + window_size, j:j + window_size]
                sobel_output[i, j] = np.sum(region * sobel_y)

        # Take absolute value
        sobel_output = np.abs(sobel_output)

        # Avoid divide-by-zero
        max_val = sobel_output.max()
        if max_val > 0:
            sobel_output = (sobel_output / max_val) * 255

        # Convert to uint8
        sobel_output = sobel_output.astype(np.uint8)

        # Convert grayscale → BGR for display_image()
        sobel_output_bgr = cv2.cvtColor(sobel_output, cv2.COLOR_GRAY2BGR)

        self.filtered_image = sobel_output_bgr
        self.display_image(sobel_output_bgr, self.filtered_img_container)

        self.filtered_img_label.configure(text=f"Sobel Y-Axis Filter - ({self.window_size.get()}) Window")

    def custom_sobelxy_filter(self):
        # Convert to grayscale
        grey_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)

        # Gaussian blur
        blurred_img = self.gaussian_blur(grey_image)

        window_size = self.get_window_size()
        # Sobel X kernel
        sobel_x = self.get_sobel_kernel(window_size, "x")

        # Sobel Y kernel
        sobel_y = self.get_sobel_kernel(window_size, "y")

        pad = window_size
        padded = np.pad(blurred_img, pad, mode='edge')

        gx = np.zeros_like(blurred_img, dtype=np.float64)
        gy = np.zeros_like(blurred_img, dtype=np.float64)

        # Compute both gradients in one loop
        for i in range(blurred_img.shape[0]):
            for j in range(blurred_img.shape[1]):
                region = padded[i:i + window_size, j:j + window_size]

                gx[i, j] = np.sum(region * sobel_x)
                gy[i, j] = np.sum(region * sobel_y)

        # Gradient magnitude
        gradient = np.sqrt(gx ** 2 + gy ** 2)

        # Normalize to 0–255
        max_val = gradient.max()
        if max_val > 0:
            gradient = (gradient / max_val) * 255

        gradient = gradient.astype(np.uint8)

        # Convert grayscale → BGR
        gradient_bgr = cv2.cvtColor(gradient, cv2.COLOR_GRAY2BGR)

        self.filtered_image = gradient_bgr
        self.display_image(gradient_bgr, self.filtered_img_container)

        self.filtered_img_label.configure(text=f"Sobel X/Y-Axis (Magnitude) - ({self.window_size.get()}) Window")

    def opencv_sobelxy_filter(self):
        window_size = self.get_window_size()

        grey_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        blurred_image = cv2.GaussianBlur(grey_image, (window_size, window_size), 0)
        sobel_x = cv2.Sobel(blurred_image, cv2.CV_16S, 1, 1, ksize=window_size)
        sobel_y = cv2.Sobel(blurred_image, cv2.CV_16S, 0, 1, ksize=window_size)

        abs_grad_x = cv2.convertScaleAbs(sobel_x)
        abs_grad_y = cv2.convertScaleAbs(sobel_y)

        sobel_xy = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

        self.filtered_image = sobel_xy
        self.display_image(sobel_xy, self.filtered_img_container)
        self.filtered_img_label.configure(text=f"OpenCV Sobel X-Y Filter - ({self.window_size.get()}) Window")

    def box_blur(self, image, window_size):
        n_rows, n_cols, n_channels = image.shape

        output_rows = n_rows - window_size + 1
        output_cols = n_cols - window_size + 1

        blurred = np.zeros((output_rows, output_cols, n_channels), dtype=np.uint8)

        for r in range(output_rows):
            for c in range(output_cols):
                # Extract window
                window = image[r:r + window_size, c:c + window_size]

                # Compute average per channel
                avg = np.mean(window, axis=(0, 1))

                blurred[r, c] = avg

        return blurred


def main():
    root = ctk.CTk()
    app = FilterApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

