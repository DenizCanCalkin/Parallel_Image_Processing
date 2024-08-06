#include <iostream>
#include <omp.h>
#include <opencv2/opencv.hpp>

// Function to resize and display an image
void resizeAndDisplay(const std::string& windowName, const cv::Mat& image, int width, int height) {
    cv::Mat resizedImage;
    cv::resize(image, resizedImage, cv::Size(width, height));
    cv::imshow(windowName, resizedImage);
}

// Function to apply the Sobel filter using parallel processing
void applySobelFilterParallel(const cv::Mat& inputImage, cv::Mat& outputImage) {
    outputImage = cv::Mat::zeros(inputImage.size(), CV_64F);

#pragma omp parallel
    {
#pragma omp for collapse(2) schedule(dynamic, 10)
        for (int row = 1; row < inputImage.rows - 1; ++row) {
            for (int col = 1; col < inputImage.cols - 1; ++col) {
                double gx = -inputImage.at<uchar>(row - 1, col - 1) - 2 * inputImage.at<uchar>(row, col - 1) - inputImage.at<uchar>(row + 1, col - 1)
                    + inputImage.at<uchar>(row - 1, col + 1) + 2 * inputImage.at<uchar>(row, col + 1) + inputImage.at<uchar>(row + 1, col + 1);
                double gy = -inputImage.at<uchar>(row - 1, col - 1) - 2 * inputImage.at<uchar>(row - 1, col) - inputImage.at<uchar>(row - 1, col + 1)
                    + inputImage.at<uchar>(row + 1, col - 1) + 2 * inputImage.at<uchar>(row + 1, col) + inputImage.at<uchar>(row + 1, col + 1);
                outputImage.at<double>(row, col) = std::sqrt(gx * gx + gy * gy);
            }
        }
    }

    cv::normalize(outputImage, outputImage, 0, 255, cv::NORM_MINMAX, CV_8U);
}

int main() {
    // Load the image
    cv::Mat img = cv::imread("C:\\Users\\calki\\Desktop\\project\\example.jpg");
    if (img.empty()) {
        std::cerr << "Could not open or find the image!\n";
        return -1;
    }

    // Convert to grayscale
    cv::Mat img_gray;
    cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);

    // Apply Gaussian blur
    cv::Mat img_blur;
    cv::GaussianBlur(img_gray, img_blur, cv::Size(3, 3), 0);

    // Apply Sobel filter in parallel
    cv::Mat edgesParallel;
    applySobelFilterParallel(img_blur, edgesParallel);

    // Display images
    int displayWidth = 800;  // Fixed width
    int displayHeight = 600; // Fixed height

    resizeAndDisplay("Original Image", img, displayWidth, displayHeight);
    resizeAndDisplay("Edges (Parallel)", edgesParallel, displayWidth, displayHeight);

    cv::waitKey(0); // Wait for a keystroke in the window
    cv::destroyAllWindows(); // Destroy all the windows

    return 0;
}
