/*
 * Copyright (c) Extreme Vision Co., Ltd.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * 算法实例中展示了如何利用ev_sdk中提供的一些库和工具源码快速完成算法的开发
 * 本算法实例中有一个管理算法配置的配置对象m_config,用于解析配置文件,并根据传入的配置字符串动态更新ROI
 * 本算法利用ji_utils.h中的工具函数实现绘图功能
 * 本算法利用三方库中的wkt工具函数实现roi的解析
 * 新添加的模型推理,跟踪等功能最好以对象成员的方式添加到算法类中,不要将过多的更能添加到同一个类中
 **/

#ifndef JI_SAMPLEALGORITHM_HPP
#define JI_SAMPLEALGORITHM_HPP
#include <string>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "ji.h"
#include "configuration.h"
#include "sample_detector.h"

using namespace std;
using namespace cv;

class SampleAlgorithm
{

public:
    
    SampleAlgorithm();
    ~SampleAlgorithm();

    /*
     * @breif 初始化算法运行资源
     * @return JiErrorCode 返回调用结果,成功返回STATUS_SUCCESS
    */    
    JiErrorCode Init();

    /*
     * @breif 去初始化，释放算法运行资源
     * @return JiErrorCode 返回调用结果,成功返回STATUS_SUCCESS
    */    
    JiErrorCode UnInit();

    /*
     * @breif 算法业务处理函数，输入分析图片，返回算法分析结果
     * @param inFrame 输入图片对象  
     * @param args 输入算法参数, json字符串
     * @param event 返回的分析结果结构体
     * @return 返回结果, STATUS_SUCCESS代表调用成功
    */    
    JiErrorCode Process(const Mat &in_frame, const char *args, JiEvent &event);

    /*
     * @breif 更新算法实例的配置
     * @param args 输入算法参数, json字符串     
     * @return 返回结果, STATUS_SUCCESS代表调用成功
    */ 
    JiErrorCode UpdateConfig(const char *args);

    /*
     * @breif 调用Process接口后,获取处理后的图像
     * @param out 返回处理后的图像结构体     
     * @param outCount 返回调用次数的计数
     * @return 返回结果, STATUS_SUCCESS代表调用成功
    */ 
    JiErrorCode GetOutFrame(JiImageInfo **out, unsigned int &out_count);

private:
    cv::Mat m_output_frame{0};    // 用于存储算法处理后的输出图像，根据ji.h的接口规范，接口实现需要负责释放该资源    
    JiImageInfo m_out_image[1];
    unsigned int m_out_count = 1;//本demo每次仅分析处理一幅图    
    Configuration m_config;     //跟配置相关的类

      
private:
    std::string m_str_last_arg;  //算法参数缓存,动态参数与缓存参数不一致时才会进行更新  
    std::string m_str_out_json;  //返回的json缓存,注意当算法实例销毁时,对应的通过算法接口获取的json字符串也将不在可用
    std::shared_ptr<SampleDetector> m_detector{nullptr}; //算法检测器实例
    cv::Ptr<cv::freetype::FreeType2> m_ft2;
};

#endif //JI_SAMPLEALGORITHM_HPP
