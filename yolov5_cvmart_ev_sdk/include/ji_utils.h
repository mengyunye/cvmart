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

#ifndef JI_UTILS
#define JI_UTILS

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <opencv2/freetype.hpp>
#include <fstream>
#include <mutex>
#define SDKLOG(b) LOG(b) << "[SDKLOG] "
#define SDKLOG_FIRST_N(b, i) LOG_FIRST_N(b, i) << "[SDKLOG] "
#define DEFAULT_FONT_PATH "/usr/local/ev_sdk/lib/fonts/NotoSansCJKsc-Regular.otf"

static std::mutex JI_LOCK;

/**
 * 获取文件大小
 *
 * @param ifs 打开的文件
 * @return 文件大小
 */
static size_t GetFileLen(std::ifstream &ifs)
{
    int orig_pos = ifs.tellg();
    ifs.seekg(0, std::fstream::end);
    size_t len = ifs.tellg();
    ifs.seekg(orig_pos);
    return len;
}

/**
 * 在图上画矩形框，并在框顶部画文字
 *
 * @param img   需要画的图
 * @param left_top_right_bottom_rect    矩形框(x, y, width, height)，其中(x, y)是左上角坐标，(width, height)是框的宽高
 * @param text  需要画的文字
 * @param rect_line_thickness 矩形框的线宽度
 * @param rect_line_type 矩形框的线类型，当值小于0时，将使用颜色填充整个矩形框
 * @param rect_color    矩形框的颜色
 * @param alpha     矩形框的透明度，范围[0,1]
 * @param font_height    字体高度
 * @param text_color 字体颜色，BGR格式
 * @param text_background    字体背景颜色，BGR格式
 */
static void DrawRectText(cv::Mat &img,
                         cv::Rect &left_top_right_bottom_rect,
                         const std::string &text,
                         int rect_line_thickness,
                         int rect_line_type,
                         cv::Scalar rect_color,
                         float rect_alpha,
                         int font_height,
                         cv::Scalar text_color,
                         cv::Scalar text_background,
                         cv::Ptr<cv::freetype::FreeType2> ft2)
{
    cv::Mat orig_data;
    if (rect_alpha < 1.0f && rect_alpha > 0.0f)
    {
        img.copyTo(orig_data);
    }
    // Draw rectangle
    cv::Point rect_left_top(left_top_right_bottom_rect.x, left_top_right_bottom_rect.y);
    cv::rectangle(img, left_top_right_bottom_rect, rect_color, rect_line_thickness, rect_line_type, 0);

    // Draw text and text background
    int baseline = 0;
    JI_LOCK.lock();

    cv::Size text_size = ft2->getTextSize(text, font_height, -1, &baseline);
    cv::Point text_left_bottom(left_top_right_bottom_rect.x, left_top_right_bottom_rect.y);
    text_left_bottom -= cv::Point(0, rect_line_thickness);
    text_left_bottom -= cv::Point(0, baseline);                                         // (left, bottom) of text
    cv::Point text_left_top(text_left_bottom.x, text_left_bottom.y - text_size.height); // (left, top) of text
    // Draw text background
    cv::rectangle(img, text_left_top, text_left_top + cv::Point(text_size.width, text_size.height + baseline), text_background,
                  cv::FILLED);
    // Draw text
    ft2->putText(img, text, text_left_bottom, font_height, text_color, -1, cv::LINE_AA, true);

    JI_LOCK.unlock();
    if (!orig_data.empty())
    { // Need to transparent drawing with alpha
        cv::addWeighted(orig_data, rect_alpha, img, (1 - rect_alpha), 0, img);
    }
}

/**
 * 在输入图img上画多边形框
 *
 * @param img   输入图
 * @param polygons  多边形数组，每个多边形由顺时针连接的点构成
 * @param color     多边形框的颜色，BGR格式
 * @param alpha     多边形框的透明度，范围[0,1]
 * @param line_type  多边形框的线类型
 * @param thickness 多边形框的宽度
 * @param is_fill    是否使用颜色填充roi区域
 */
static void DrawPolygon(cv::Mat &img,
                        std::vector<std::vector<cv::Point>> polygons,
                        const cv::Scalar &color,
                        float alpha,
                        int line_type,
                        int thickness,
                        bool is_fill)
{
    cv::Mat orig_data;
    bool fill = (is_fill && alpha < 1.0f && alpha > 0.0f);
    if (fill)
    {
        img.copyTo(orig_data);
    }
    for (size_t i = 0; i < polygons.size(); i++)
    {
        const cv::Point *pPoint = &polygons[i][0];
        int n = (int)polygons[i].size();
        if (fill)
        {
            cv::fillPoly(img, &pPoint, &n, 1, color, line_type);
        }
        else
        {
            cv::polylines(img, &pPoint, &n, 1, true, color, thickness, line_type);
        }
    }
    if (!orig_data.empty())
    { // Transparent drawing
        cv::addWeighted(orig_data, alpha, img, (1 - alpha), 0, img);
    }
}

/**
 * 在img上画文字text
 *
 * @param img   输入图
 * @param text  文字
 * @param font_height    文字大小
 * @param foreground_color   文字颜色，BGR格式
 * @param background_color   文字背景颜色，BGR格式
 * @param leftTop   所画文字的左上顶点所在位置
 */
static void DrawText(cv::Mat &img,
                     const std::string &text,
                     int font_height,
                     const cv::Scalar &foreground_color,
                     const cv::Scalar &background_color,
                     const cv::Point &left_top_shift,
                     cv::Ptr<cv::freetype::FreeType2> ft2)
{
    if (text.empty())
    {
        printf("text cannot be empty!\n");
        return;
    }

    int baseline = 0;
    JI_LOCK.lock();
    cv::Size text_size = ft2->getTextSize(text, font_height, -1, &baseline);
    cv::Point text_left_bottom(0, text_size.height);
    text_left_bottom -= cv::Point(0, baseline);                                         // (left, bottom) of text
    cv::Point text_left_top(text_left_bottom.x, text_left_bottom.y - text_size.height); // (left, top) of text
    // Draw text background
    text_left_top += left_top_shift;
    cv::rectangle(img, text_left_top, text_left_top + cv::Point(text_size.width, text_size.height + baseline), background_color,
                  cv::FILLED);
    text_left_bottom += left_top_shift;
    ft2->putText(img, text, text_left_bottom, font_height, foreground_color, -1, cv::LINE_AA, true);
    JI_LOCK.unlock();
}

#endif // JI_UTILS
