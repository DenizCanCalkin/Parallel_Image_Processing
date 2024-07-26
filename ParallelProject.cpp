#include <iostream>
#include <omp.h>
#include <opencv2/opencv.hpp>

void resizeAndDisplay(const std::string& windowName, const cv::Mat& image, int width, int height) {
    cv::Mat resizedImage;
    cv::resize(image, resizedImage, cv::Size(width, height));
    cv::imshow(windowName, resizedImage);
}

void applySobelFilterSerial(const cv::Mat& inputImage, cv::Mat& grad_x, cv::Mat& grad_y, cv::Mat& outputImage) {
    cv::Sobel(inputImage, grad_x, CV_64F, 1, 0, 3);
    cv::Sobel(inputImage, grad_y, CV_64F, 0, 1, 3);
    cv::magnitude(grad_x, grad_y, outputImage);
    cv::normalize(outputImage, outputImage, 0, 255, cv::NORM_MINMAX, CV_8U);
}

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

void applySobelFilterParallelOptimized(const cv::Mat& inputImage, cv::Mat& outputImage) {
    outputImage = cv::Mat::zeros(inputImage.size(), CV_64F);

#pragma omp parallel for schedule(dynamic, 10)
    for (int row = 1; row < inputImage.rows - 1; ++row) {
        for (int col = 1; col < inputImage.cols - 1; ++col) {
            double gx = -inputImage.at<uchar>(row - 1, col - 1) - 2 * inputImage.at<uchar>(row, col - 1) - inputImage.at<uchar>(row + 1, col - 1)
                + inputImage.at<uchar>(row - 1, col + 1) + 2 * inputImage.at<uchar>(row, col + 1) + inputImage.at<uchar>(row + 1, col + 1);
            double gy = -inputImage.at<uchar>(row - 1, col - 1) - 2 * inputImage.at<uchar>(row - 1, col) - inputImage.at<uchar>(row - 1, col + 1)
                + inputImage.at<uchar>(row + 1, col - 1) + 2 * inputImage.at<uchar>(row + 1, col) + inputImage.at<uchar>(row + 1, col + 1);
            outputImage.at<double>(row, col) = std::sqrt(gx * gx + gy * gy);
        }
    }

    cv::normalize(outputImage, outputImage, 0, 255, cv::NORM_MINMAX, CV_8U);
}

int main() {
    cv::Mat img = cv::imread("C:\\Users\\calki\\Desktop\\Project\\example.jpg");
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

    // Apply Sobel filter serially
    cv::Mat grad_x, grad_y, edgesSerial;
    applySobelFilterSerial(img_blur, grad_x, grad_y, edgesSerial);

    // Apply Sobel filter in parallel
    cv::Mat edgesParallel;
    applySobelFilterParallel(img_blur, edgesParallel);

    // Apply Sobel filter in parallel (optimized)
    cv::Mat edgesParallelOptimized;
    applySobelFilterParallelOptimized(img_blur, edgesParallelOptimized);

    // Compare results
    cv::Mat diff;
    cv::absdiff(edgesSerial, edgesParallel, diff);
    double maxDiff = cv::norm(diff, cv::NORM_INF);

    if (maxDiff < 1e-6) {
        std::cout << "The parallel implementation is correct!\n";
    }
    else {
        std::cout << "There are differences between the serial and parallel implementations.\n";
    }

    // Compare results for optimized version
    cv::Mat diffOptimized;
    cv::absdiff(edgesSerial, edgesParallelOptimized, diffOptimized);
    double maxDiffOptimized = cv::norm(diffOptimized, cv::NORM_INF);

    if (maxDiffOptimized < 1e-6) {
        std::cout << "The optimized parallel implementation is correct!\n";
    }
    else {
        std::cout << "There are differences between the serial and optimized parallel implementations.\n";
    }

    // Display images
    int displayWidth = 800;  // Fixed width
    int displayHeight = 600; // Fixed height

    resizeAndDisplay("Original Image", img, displayWidth, displayHeight);
    resizeAndDisplay("Edges (Serial)", edgesSerial, displayWidth, displayHeight);
    resizeAndDisplay("Edges (Parallel)", edgesParallel, displayWidth, displayHeight);
    resizeAndDisplay("Edges (Parallel Optimized)", edgesParallelOptimized, displayWidth, displayHeight);

    cv::waitKey(0);
    cv::destroyAllWindows();
    return 0;
}
