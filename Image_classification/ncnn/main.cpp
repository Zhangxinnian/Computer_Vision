#include "include/ncnn/net.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <vector>


static int recognition_resnet(const cv::Mat& bgr, std::vector<float>& cls_scores)
{
    ncnn::Net resnet;
    resnet.opt.use_vulkan_compute = true;
    resnet.load_param("/home/cj1/CLionProjects/Image_classification/ncnn_model/cifar_resnet-0000.param");
    resnet.load_model("/home/cj1/CLionProjects/Image_classification/ncnn_model/cifar_resnet-0000.bin");

    //const int target_size = 224;
    const int target_size = 32;
    int img_w = bgr.cols;
    int img_h = bgr.rows;
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR, bgr.cols, bgr.rows, target_size, target_size);

    //const float mean_val[3] = {0.485, 0.456, 0.406};
    const float mean_val[3] = {0.4914, 0.4822, 0.4465};
    //const float norm_val[3] = {0.229, 0.224, 0.225};
    const float norm_val[3] = {0.2023, 0.1994, 0.2010};
    in.substract_mean_normalize(mean_val, norm_val);
    ncnn::Extractor ex = resnet.create_extractor();
    ex.input("data", in);
    ncnn::Mat out;
    ex.extract("cifarresnetv10_dense0_fwd", out);
//    {
//        ncnn::Layer* softmax = ncnn::create_layer("Softmax");
//        ncnn::ParamDict pd;
//        softmax->load_param(pd);
//        softmax->forward_inplace(out, resnet.opt);
//        delete softmax;
//    }
//    out = out.reshape(out.w * out.h * out.c);
    cls_scores.resize(out.w);
    for (int j = 0; j < out.w ; j++) {
        cls_scores[j] = out[j];
    }

    return 0;
}

static int print_topk(const cv::Mat& bgr, const std::vector<float>& cls_scores, int topk)
{
    static const char* class_names[] = {
            "airplane", "qutomobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"
    };
    cv::Mat image = bgr.clone();
    int size = cls_scores.size();
    std::vector<std::pair<float, int> > vec;
    vec.resize(size);
    for (int i = 0; i < size; i++)
    {
        vec[i] = std::make_pair(cls_scores[i], i);
    }

    std::partial_sort(vec.begin(), vec.begin() + topk, vec.end(),
                      std::greater<std::pair<float, int> >());

    // print topk and score
    for (int i = 0; i < topk; i++)
    {
        float score = vec[i].first;
        int index = vec[i].second;
        fprintf(stderr, "%d = %f\n", index, score );
        int baseLine = 0;
    }
    cv::putText(image, class_names[vec[0].second],  cv::Point(10,10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,0,0));
    cv::imshow("image", image);
    cv::waitKey(0);

    return 0;
}



int main(int argc, char** argv)
{
//    if(argc != 2)
//    {
//        fprintf(stderr, "Usage: %s [imagepath]\n", argv[0]);
//        return -1;
//    }
    //const char* imagepath = argv[1];
    const char * imagepath = "/home/cj1/CLionProjects/Image_classification/image/dog.jpg";
    cv::Mat m = cv::imread(imagepath, 1);
    if (m.empty())
    {
        //fprintf(stderr, "cv::imread %s failed\n",imagepath);
        printf("cv::imread is failed\n");
        return -1;
    }

    std::vector<float> cls_scores;
    recognition_resnet(m,cls_scores);
    print_topk(m, cls_scores, 3);
    return 0;
}

