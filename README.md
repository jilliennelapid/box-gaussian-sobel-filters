# Computer Vision - Assignment 2

Using custom implementation and the OpenCV library, blurring and image detection were applied to images via the Box Filter, Gaussian Filter, and Sobel Filter.

For the OpenCV functions, blurring of images was achieved with `boxFilter()` and `GaussianBlur()` where edge detection was achieved with `Sobel()`.

For Sobel Filtering, the custom functions apply Sobel in either the x-axis, y-axis, or both axes.
Sobel Filtering with OpenCV was applied in both axes.

The convolving window (kernel) size for these operations was selectable between sizes of 3x3 and 5x5.

____

An interface was created to allow for the loading of image, selection of kernel size, and selection of filtering method.
The GUI was created with Tom Schiamansky's [`customtkinter`](https://customtkinter.tomschimansky.com) library.
