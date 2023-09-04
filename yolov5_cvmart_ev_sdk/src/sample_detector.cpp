#include <sys/stat.h>
#include <fstream>
#include <glog/logging.h>

#include "sample_detector.h"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "ji_utils.h"


using namespace ev;
namespace ev
{
namespace vision
{

float IOU(const cv::Rect &b1, const cv::Rect &b2)
{
    auto intersec = b1 & b2;
    return static_cast<float>(intersec.area()) / (b1.area() + b2.area() - intersec.area());
}

void NMS(std::vector<BoxInfo> &objects, float iou_thresh)
{

    
    auto cmp_lammda = [](const BoxInfo &b1, const BoxInfo &b2)
    { return b1.score > b2.score; };
    std::sort(objects.begin(), objects.end(), cmp_lammda);
    for (int i = 0; i < objects.size(); ++i)
    {
        if(objects[i].score  < 0.000001){
            continue; //已经删除，跳过
        }
        for (int j = i + 1; j < objects.size(); ++j)
        {
            if(objects[i].label != objects[j].label){//不同类别，跳过
                continue;
            }
            cv::Rect rect1 = cv::Rect{objects[i].x1, objects[i].y1, objects[i].x2 - objects[i].x1, objects[i].y2 - objects[i].y1};
            cv::Rect rect2 = cv::Rect{objects[j].x1, objects[j].y1, objects[j].x2 - objects[i].x1, objects[j].y2 - objects[j].y1};
            if (IOU(rect1, rect2) > iou_thresh)
            {
                objects[j].score = 0.f;
                SDKLOG(INFO) << "erase id：" << j;
            }
        }
    }
    auto iter = objects.begin();
    while (iter != objects.end())
    {
        if (iter->score < 0.000001)
        {
            iter = objects.erase(iter);
        }
        else
        {
            ++iter;
        }
    }
}
}//namespace vision
}//namepspace ev
SampleDetector::SampleDetector()
{
}

bool SampleDetector::Init(float thresh, const std::string &uuid)
{
    m_thresh = thresh;//传入后处理阈值
    m_uuid = uuid;//传入的模型uuid
}

bool SampleDetector::UnInit()
{
}

SampleDetector::~SampleDetector()
{
    UnInit();
}

bool SampleDetector::ProcessImage(const cv::Mat &img, std::vector<ev::vision::BoxInfo> &det_results)
{
    det_results.clear();
    cv::Mat cv_in_mat1;
    //前处理
    m_preprocessor.Run(const_cast<cv::Mat&>(img), cv_in_mat1, 480);

    //准备输入数据
    EVModelData in;
    EVModelData out;
    EVMatData in_mat;

    in.desc = NULL;
    in.mat = &in_mat;

    in.mat_num = 1; // 输入图像数量,也可以是多张;如果是多张,则in.mat为数组指针

    in_mat.data = cv_in_mat1.data;
    in_mat.data_size = cv_in_mat1.cols * cv_in_mat1.rows * 3 * 4;
    in_mat.width = cv_in_mat1.cols;
    in_mat.height = cv_in_mat1.rows;
    in_mat.aligned_width = cv_in_mat1.cols;
    in_mat.aligned_height = cv_in_mat1.rows;
    in_mat.channel = 3;
    in_mat.loc = EV_DATA_HOST;
    in_mat.type = EV_UINT8;

    //执行推理
    std::lock_guard<std::mutex> lock_guard(m_mutex);//用于多线程时,线程安全
    EVDeploy::GetModel().RunInfer(m_uuid, &in, &out);
    SDKLOG(INFO) << "RunInfer done";
    // 输出的数量由out.mat_num指示,输出的数据封装在out.mat中,如果是多个输出,则out.mat为指向多个输出的指针,
    // 每一个输出的维度信息由out.mat[i]->dims指示
    // 每一个输出的名称信息由out.mat[i]->desc指示
    for (int j = 0; j < out.mat_num; ++j)
    {
        SDKLOG(INFO) << "output name: " << out.mat[j].desc;
        for (int k = 0; k < out.mat[j].dims.size(); ++k)
        {
            SDKLOG(INFO) << "dims " << k << ":" << out.mat[j].dims[k];
        }
    }

    //后处理   
    float scale = m_preprocessor.GetScale();
    SampleDetector::YOLOv5Postprocessor(out.mat, det_results, scale, m_thresh, img.cols, img.rows);

    // 注意释放out.mat,否则会有内存泄露!!!!
    if (out.mat)
    {
        delete[] out.mat;
    }

    return true;
}
EVStatus SampleDetector::YOLOv5Postprocessor(EVMatData *out, std::vector<ev::vision::BoxInfo> &objects, float scale, float thresh, const int img_w, const int img_h)
{

    REC_TIME(t0);
    std::vector<ev::vision::BoxInfo> proposals;

    // need to know the net output shape
    int box_num = out->dims[1];
    // need to know the output class num
    int class_num = out->dims[2] - 5;
    float *buffer = (float *)out->data;
    for (int i = 0; i < box_num; ++i)
    {
        int index = i * (class_num + 5);
        // EVLOG(INFO) << buffer[index+4];
        if (buffer[index + 4] > thresh) // det_thresh 0.2
        {
            float x = buffer[index] / scale;
            float y = buffer[index + 1] / scale;
            float w = buffer[index + 2] / scale;
            float h = buffer[index + 3] / scale;
            float *max_cls_pos = std::max_element(buffer + index + 5, buffer + index + 5 + class_num);

            if ((*max_cls_pos) * buffer[index + 4] > thresh)
            {
                cv::Rect box{x - w / 2, y - h / 2, w, h};
                box = box & cv::Rect(0, 0, img_w - 1, img_h - 1);
                if (box.area() > 0)
                {
                   ev::vision::BoxInfo box_info = {box.x, box.y, box.x + box.width, box.y + box.height, (*max_cls_pos) * buffer[index + 4], max_cls_pos - (buffer + index + 5)};
                    objects.push_back(box_info);
                }
            }
        }
    }

    NMS(objects, 0.45);
    REC_TIME(t1);
    EVLOG(INFO) << "YOLOv5Postprocessor run time(ms):" << RUN_TIME(t1 - t0);
    return EV_SUCCESS;
}