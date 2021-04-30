#include <iostream>
#include <ctime>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>

using namespace cv;
using namespace std;

int main() {
    try {
        clock_t begin = clock();
        Mat dstHost;
        String filename = "../data/1.png";
        Mat srcHost = imread(filename, IMREAD_GRAYSCALE);

        for (int i = 0; i < 100000; i++) {
            cuda::GpuMat dst, src;
            src.upload(srcHost);
            cuda::threshold(src, src, 128.0, 255.0, THRESH_BINARY);
            cuda::bilateralFilter(src, dst, 3, 1, 1);
            dst.download(dstHost);
        }
        clock_t end = clock();
        cout << "GPU: " << double(end - begin) / CLOCKS_PER_SEC << endl;
        // GPU: 64.2302

        begin = clock();
        for (int i = 0; i < 100000; i++) {
            Mat dst;
            threshold(srcHost, srcHost, 128.0, 255.0, THRESH_BINARY);
            bilateralFilter(srcHost, dst, 3, 1, 1);
        }

        end = clock();
        cout << "CPU: " << double(end - begin) / CLOCKS_PER_SEC << endl;
        // CPU: 973.586

        // imshow("Result", dstHost);
        // waitKey();

    } catch (const Exception &ex) {
        cout << "Error: " << ex.what() << endl;
    }
}