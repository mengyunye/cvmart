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

#ifndef JI_ALGORITHM_CONFIGURATION
#define JI_ALGORITHM_CONFIGURATION
#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <opencv2/opencv.hpp>

#include "wkt_parser.h"
#include "ji_utils.h"
#include "evdeploy/deploy.h"

using namespace ev;

#define BGRA_CHANNEL_SIZE 4

typedef struct
{
    // 与自定义算法密切相关的配置参数，可封装在该结构体中
    double thresh;
    std::string uuid;
} AlgoConfig;

/**
 * @brief 存储和更新配置参数
 */
struct Configuration
{
    private:
    ev::Value m_config_value;
    public:    
    // 算法与画图的可配置参数及其默认值
    // 1. roi配置
    std::vector<cv::Rect> current_roi_rects;           // 多边形roi区域对应的矩形区域（roi可能是多个）
    std::vector<VectorPoint> current_roi_orig_polygons; // 原始的多边形roi区域（roi可能是多个）
    std::vector<std::string> orig_roi_args;            //原始的多边形roi区域对应的字符串
    cv::Size current_in_frame_size{0, 0};               // 当前处理帧的尺寸
    // 2. 与ROI显示相关的配置
    bool draw_roi_area = false;                    // 是否画ROI
    cv::Scalar roi_color = {120, 120, 120, 1.0f}; // ROI框的颜色
    int roi_line_thickness = 4;                    // ROI框的粗细
    bool roi_fill = false;                        // 是否使用颜色填充ROI区域
    bool draw_result = true;                      // 是否画识别结果
    bool draw_confidence = false;                 // 是否画置信度
    // --------------------------------- 通常需要根据需要修改 START -----------------------------------------
    // 3. 算法配置参数
    AlgoConfig algo_config = {0.4, "000000"}; // 默认的算法配置
    // 4. 与报警信息相关的配置
    std::string language = "en";                             // 所显示文字的默认语言
    int target_rect_line_thickness = 4;                         // 目标框粗细
    std::map<std::string, std::vector<std::string> > target_rect_text_map = { {"en",{"person"}}, {"zh", {"行人"}}};// 检测目标框顶部文字
    cv::Scalar target_rect_color = {0, 255, 0, 1.0f}; // 检测框`mark`的颜色
    cv::Scalar text_foreground_color = {0, 0, 0, 0};          // 检测框顶部文字的颜色
    cv::Scalar text_background_color = {255, 255, 255, 0};    // 检测框顶部文字的背景颜色
    int target_text_height = 30;                      // 目标框顶部字体大小

    bool draw_warning_text = true;
    int warning_text_size = 40;                             // 画到图上的报警文字大小
    std::map<std::string, std::string> warning_text_map = { {"en", "WARNING!"}, {"zh", "警告"}};// 画到图上的报警文字
    cv::Scalar warning_text_foreground_color = {255, 255, 255, 0}; // 报警文字颜色
    cv::Scalar warning_text_background_color = {0, 0, 255, 0};     // 报警文字背景颜色
    cv::Point warning_text_left_top{0, 0};            // 报警文字左上角位置
    // --------------------------------- 通常需要根据需要修改 END -------------------------------------------
    //解析数值类型的配置
    template <typename T>
    bool CheckAndUpdateNumber(const std::string& key, T &val)
    {
      return  m_config_value.isMember(key) && m_config_value[key].isNumeric() && ( (val = m_config_value[key].asDouble()) || true); 
    }
    //解析字符串类型配置的函数
    bool CheckAndUpdateStr(const std::string& key, std::string &val)
    {
        return m_config_value.isMember(key) && m_config_value[key].isString() &&  (val = m_config_value[key].asString()).size(); 
    }
    //解析字符串数组类型配置的函数
    bool CheckAndUpdateVecStr(const std::string& key, std::vector<std::string> &val)
    {
        if( m_config_value.isMember(key) && m_config_value[key].isArray() ) 
        {
            val.resize(m_config_value[key].size());
            for(int i = 0; i <  m_config_value[key].size(); ++i)
            {
                val[i] = m_config_value[key][i].asString();
            }
        }
        return true;
    }
    //解析bool类型配置的函数
    bool CheckAndUpdateBool(const std::string& key, bool &val)
    {
        return m_config_value.isMember(key) && m_config_value[key].isBool() && ( (val = m_config_value[key].asBool()) || true); 
    }
    //解析颜色配置的函数
    bool CheckAndUpdateColor(const std::string& key, cv::Scalar &color)
    {
        if(m_config_value.isMember(key) && m_config_value[key].isArray() && m_config_value[key].size() == BGRA_CHANNEL_SIZE)
        {
            for (int i = 0; i < BGRA_CHANNEL_SIZE; ++i)
            {
                color[i] = m_config_value[key][i].asDouble();
            }
            return true;
        }
        return true;
    }
    //解析点配置的函数
    bool CheckAndUpdatePoint(const std::string& key, cv::Point &point)
    {
        if(m_config_value.isMember(key) && m_config_value[key].isArray() && m_config_value[key].size() == 2 && m_config_value[key][0].isNumeric() && m_config_value[key][1].isNumeric())
        {
            point = cv::Point(m_config_value[key][0].asDouble(), m_config_value[key][1].asDouble());
            return true;
        }
        return true;
    }

    /**
     * @brief 解析json格式的配置参数,是开发者需要重点关注和修改的地方！！！     
     * @param[in] configStr json格式的配置参数字符串
     * @return 当前参数解析后，生成的算法相关配置参数
     */
    void ParseAndUpdateArgs(const char *conf_str)
    {
        if (conf_str == nullptr)
        {
            SDKLOG(INFO) << "Input is none";
            return;
        }        
        ev::Reader reader;
        if( !reader.parse(conf_str, m_config_value) )
        {
            SDKLOG(ERROR) << "failed to parse config " << conf_str;
        }
        CheckAndUpdateBool("draw_roi_area", draw_roi_area);
        CheckAndUpdateNumber("thresh", algo_config.thresh);
        CheckAndUpdateStr("uuid", algo_config.uuid);            
        CheckAndUpdateNumber("roi_line_thickness", roi_line_thickness);
        CheckAndUpdateBool("roi_fill", roi_fill);
        CheckAndUpdateStr("language", language);
        CheckAndUpdateBool("draw_result", draw_result);
        CheckAndUpdateBool("draw_confidence", draw_confidence);
        CheckAndUpdateVecStr("mark_text_en", target_rect_text_map["en"]);        
        CheckAndUpdateVecStr("mark_text_zh", target_rect_text_map["zh"]);        
        CheckAndUpdateColor("roi_color", roi_color);
        CheckAndUpdateColor("object_text_color", text_foreground_color);
        CheckAndUpdateColor("object_text_bg_color", text_background_color);
        CheckAndUpdateColor("target_rect_color", target_rect_color);
        CheckAndUpdateNumber("object_rect_line_thickness", target_rect_line_thickness);
        CheckAndUpdateNumber("object_text_size", target_text_height);
        CheckAndUpdateBool("draw_warning_text", draw_warning_text);
        CheckAndUpdateNumber("warning_text_size", warning_text_size);
        CheckAndUpdateStr("warning_text_en", warning_text_map["en"]);  
        CheckAndUpdateStr("warning_text_zh", warning_text_map["zh"]);  
        CheckAndUpdateColor("warning_text_color", warning_text_foreground_color);
        CheckAndUpdateColor("warning_text_bg_color", warning_text_background_color);       
        CheckAndUpdatePoint("warning_text_left_top", warning_text_left_top);
        std::vector<std::string> roiStrs;
        if(m_config_value.isMember("polygon_1") && m_config_value["polygon_1"].isArray() && m_config_value["polygon_1"].size() )
        {
            for (int i = 0; i < m_config_value["polygon_1"].size(); ++i)
            {                
                if(m_config_value["polygon_1"][i].isString())
                {
                    roiStrs.emplace_back(m_config_value["polygon_1"][i].asString());
                }
            }
        }        
        if (!roiStrs.empty())
        {
            orig_roi_args = roiStrs;
            UpdateROIInfo(current_in_frame_size.width, current_in_frame_size.height);//根据当前输入图像帧的大小更新roi参数
        }
                
        return;
    }
    /**
     * @brief 当输入图片尺寸变更时，更新ROI
     **/
    void UpdateROIInfo(int new_width, int new_height)
    {
        current_in_frame_size.width = new_width;
        current_in_frame_size.height = new_height;
        current_roi_orig_polygons.clear();
        current_roi_rects.clear();

        VectorPoint current_frame_polygon;
        current_frame_polygon.emplace_back(cv::Point(0, 0));
        current_frame_polygon.emplace_back(cv::Point(current_in_frame_size.width, 0));
        current_frame_polygon.emplace_back(cv::Point(current_in_frame_size.width, current_in_frame_size.height));
        current_frame_polygon.emplace_back(cv::Point(0, current_in_frame_size.height));

        WKTParser wkt_parser(cv::Size(new_width, new_height));
        for (auto &roi_str : orig_roi_args)
        {
            SDKLOG(INFO) << "parsing roi:" << roi_str;
            VectorPoint polygon;
            wkt_parser.ParsePolygon(roi_str, &polygon);
            bool is_polygon_valid = true;
            for (auto &point : polygon)
            {
                if (!wkt_parser.InPolygon(current_frame_polygon, point))
                {
                    SDKLOG(ERROR) << "point " << point << " not in polygon!";
                    is_polygon_valid = false;
                    break;
                }
            }
            if (!is_polygon_valid || polygon.empty())
            {
                SDKLOG(ERROR) << "roi `" << roi_str << "` not valid! skipped!";
                continue;
            }
            current_roi_orig_polygons.emplace_back(polygon);
        }
        if (current_roi_orig_polygons.empty())
        {
            current_roi_orig_polygons.emplace_back(current_frame_polygon);
            SDKLOG(WARNING) << "Using the whole image as roi!";
        }

        for (auto &roiPolygon : current_roi_orig_polygons)
        {
            cv::Rect rect;
            wkt_parser.Polygon2Rect(roiPolygon, rect);
            current_roi_rects.emplace_back(rect);
        }
    }
};
#endif
