#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    std::string imagePath = "C:/Users/rafae/Desktop/4_kurs/4_kurs/kram/lab4/777.jpg";

    cv::Mat img = cv::imread(imagePath);
    if (img.empty()) {
        std::cerr << "Не удалось открыть файл: " << imagePath << std::endl;
        return -1;
    }

    cv::Mat gray, blur, edges;

    
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

    
    int ksize = 5;
    double sigma = 1.0;
    cv::GaussianBlur(gray, blur, cv::Size(ksize, ksize), sigma);

    
    double lowThresh = 40;   // нижний порог
    double highThresh = 120; // верхний порог
    cv::Canny(blur, edges, lowThresh, highThresh);

    
    cv::imshow("Original", img);
    cv::imshow("Gray + Gaussian blur", blur);
    cv::imshow("Canny edges (C++)", edges);
    cv::waitKey(0);

    return 0;
}
