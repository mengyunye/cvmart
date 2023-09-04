#include <cstdlib>
#include <cstring>
#include <iostream>
#include <fstream>
#include <glog/logging.h>

#include "evdeploy/deploy.h"
#include "ji_utils.h"
#include "sample_algorithm.h"

#define JSON_ALERT_FLAG_KEY ("is_alert")
#define JSON_ALERT_FLAG_TRUE true
#define JSON_ALERT_FLAG_FALSE false

using namespace ev;

SampleAlgorithm::SampleAlgorithm()
{
}

SampleAlgorithm::~SampleAlgorithm()
{
    UnInit();
}

JiErrorCode SampleAlgorithm::Init()
{
    // 从默认的配置文件读取相关配置参数
    const char *config_file = "/usr/local/ev_sdk/config/algo_config.json";
    SDKLOG(INFO) << "Parsing configuration file: " << config_file;
    std::ifstream conf(config_file);
    if (conf.is_open())
    {
        size_t len = GetFileLen(conf);
        char *conf_str = new char[len + 1];
        conf.read(conf_str, len);
        conf_str[len] = '\0';
        SDKLOG(INFO) << "Configs:" << conf_str;
        m_config.ParseAndUpdateArgs(conf_str);
        delete[] conf_str;
        conf.close();
    }

    m_detector = std::make_shared<SampleDetector>();
    // 注意uuid,可以从algo_config.json中去配置获取,也可以直接给出
    m_detector->Init(m_config.algo_config.thresh, m_config.algo_config.uuid);
    m_ft2 = cv::freetype::createFreeType2();
    m_ft2->loadFontData(DEFAULT_FONT_PATH, 0);
    return JISDK_RET_SUCCEED;
}

JiErrorCode SampleAlgorithm::UnInit()
{
    if (m_detector.get() != nullptr)
    {
        SDKLOG(INFO) << "uninit";
        m_detector->UnInit();
    }
    return JISDK_RET_SUCCEED;
}

JiErrorCode SampleAlgorithm::UpdateConfig(const char *args)
{
    if (args == nullptr)
    {
        SDKLOG(ERROR) << "m_config string is null ";
        return JISDK_RET_FAILED;
    }
    m_config.ParseAndUpdateArgs(args);
    return JISDK_RET_SUCCEED;
}

JiErrorCode SampleAlgorithm::GetOutFrame(JiImageInfo **out, unsigned int &out_count)
{
    out_count = m_out_count;

    m_out_image[0].nWidth = m_output_frame.cols;
    m_out_image[0].nHeight = m_output_frame.rows;
    m_out_image[0].nFormat = JI_IMAGE_TYPE_BGR;
    m_out_image[0].nDataType = JI_UNSIGNED_CHAR;
    m_out_image[0].nWidthStride = m_output_frame.step;
    m_out_image[0].pData = m_output_frame.data;

    *out = m_out_image;
    return JISDK_RET_SUCCEED;
}

JiErrorCode SampleAlgorithm::Process(const cv::Mat &in_frame, const char *args, JiEvent &event)
{
    // 输入图片为空的时候直接返回错误
    if (in_frame.empty())
    {
        SDKLOG(ERROR) << "Invalid input!";
        return JISDK_RET_FAILED;
    }

    // 由于roi配置是归一化的坐标,所以输出图片的大小改变时,需要更新ROI的配置
    if (in_frame.cols != m_config.current_in_frame_size.width || in_frame.rows != m_config.current_in_frame_size.height)
    {
        SDKLOG(INFO) << "Update ROI Info...";
        m_config.UpdateROIInfo(in_frame.cols, in_frame.rows);
    }

    // 如果输入的参数不为空且与上一次的参数不完全一致,需要调用更新配置的接口
    if (args != nullptr && m_str_last_arg != args)
    {
        m_str_last_arg = args;
        SDKLOG(INFO) << "Update args:" << args;
        m_config.ParseAndUpdateArgs(args);
    }

    // 针对整张图进行推理,获取所有的检测目标,并过滤出在ROI内的目标
    std::vector<ev::vision::BoxInfo> detected_objects;
    // std::vector<ev::vision::BoxInfo> valid_targets;
    // 算法处理
    cv::Mat img = in_frame.clone();

    m_detector->ProcessImage(img, detected_objects);

    // 过滤出行人
    // for (auto iter = detected_objects.begin(); iter != detected_objects.end();)
    // {
    //     SDKLOG(INFO) << "label:" << iter->label;
    //     if (iter->label == 0)
    //     {
    //         iter++;
    //     }
    //     else
    //     {
    //         iter = detected_objects.erase(iter);
    //     }
    // }

    // for (auto &obj : detected_objects)
    // {
    //     for (auto &roi : m_config.current_roi_orig_polygons)
    //     {
    //         int mid_x = (obj.x1 + obj.x2) / 2;
    //         int mid_y = (obj.y1 + obj.y2) / 2;
    //         // 当检测的目标的中心点在ROI内的话，就视为闯入ROI的有效目标
    //         if (WKTParser::InPolygon(roi, cv::Point(mid_x, mid_y)))
    //         {
    //             valid_targets.emplace_back(obj);
    //         }
    //     }
    // }
    // SDKLOG_FIRST_N(INFO, 5) << "detected targets : " << detected_objects.size() << " valid targets :  " << valid_targets.size();
    REC_TIME(t0);
    // 此处示例业务逻辑：当算法检测到有行人闯入时，就报警
    bool is_need_alert = false; // 是否需要报警
    // 创建输出图
    // in_frame.copyTo(m_output_frame);
    // // 画ROI区域
    // if (m_config.draw_roi_area && !m_config.current_roi_orig_polygons.empty())
    // {
    //     DrawPolygon(m_output_frame,
    //                 m_config.current_roi_orig_polygons,
    //                 cv::Scalar(m_config.roi_color[0], m_config.roi_color[1], m_config.roi_color[2]),
    //                 m_config.roi_color[3],
    //                 cv::LINE_AA,
    //                 m_config.roi_line_thickness,
    //                 m_config.roi_fill);
    // }
    // 判断是否要要报警
//     if (valid_targets.size() > 0)
//     {
//         is_need_alert = true;
//     }
//     // 并将检测到的在ROI内部的目标画到图上
//     for (auto &object : valid_targets)
//     {
//         if (m_config.draw_result)
//         {
//             std::stringstream ss;
//             ss << (object.label > m_config.target_rect_text_map[m_config.language].size() - 1 ? "" : m_config.target_rect_text_map[m_config.language][object.label]);
//             if (m_config.draw_confidence)
//             {
//                 ss.precision(0);
//                 ss << std::fixed << (object.label > m_config.target_rect_text_map[m_config.language].size() - 1 ? "" : ": ") << object.score * 100 << "%";
//             }
//             cv::Rect rect = cv::Rect{object.x1, object.y1, object.x2 - object.x1, object.y2 - object.y1};
//             DrawRectText(m_output_frame,
//                          rect, ss.str(),
//                          m_config.target_rect_line_thickness,
//                          cv::LINE_AA,
//                          cv::Scalar(m_config.target_rect_color[0], m_config.target_rect_color[1], m_config.target_rect_color[2]),
//                          m_config.target_rect_color[3],
//                          m_config.target_text_height,
//                          cv::Scalar(m_config.text_foreground_color[0], m_config.text_foreground_color[1], m_config.text_foreground_color[2]),
//                          cv::Scalar(m_config.text_background_color[0], m_config.text_background_color[1], m_config.text_background_color[2]),
//                          m_ft2);
//         }
//     }

//     if (is_need_alert && m_config.draw_warning_text)
//     {
//         DrawText(m_output_frame,
//                  m_config.warning_text_map[m_config.language],
//                  m_config.warning_text_size,
//                  cv::Scalar(m_config.warning_text_foreground_color[0], m_config.warning_text_foreground_color[1], m_config.warning_text_foreground_color[2]),
//                  cv::Scalar(m_config.warning_text_background_color[0], m_config.warning_text_background_color[1], m_config.warning_text_background_color[2]),
//                  m_config.warning_text_left_top,
//                  m_ft2);
//     }

    // 将结果封装成json字符串
    bool json_alert_code = JSON_ALERT_FLAG_FALSE;
    if (is_need_alert)
    {
        json_alert_code = JSON_ALERT_FLAG_TRUE;
    }
    ev::Value json_root;
    ev::Value json_algo_value;
    ev::Value json_detect_value;

    json_algo_value[JSON_ALERT_FLAG_KEY] = json_alert_code;
    json_algo_value["target_info"].resize(0);
    // for (auto &obj : valid_targets)
    // {
    //     ev::Value tmp_value;
    //     tmp_value["x"] = int(obj.x1);
    //     tmp_value["y"] = int(obj.y1);
    //     tmp_value["width"] = int(obj.x2 - obj.x1);
    //     tmp_value["height"] = int(obj.y2 - obj.y1);
    //     tmp_value["name"] = (obj.label > m_config.target_rect_text_map[m_config.language].size() - 1 ? "obj" : m_config.target_rect_text_map[m_config.language][obj.label]);
    //     tmp_value["confidence"] = obj.score;
    //     json_algo_value["target_info"].append(tmp_value);
    // }
    // json_root["algorithm_data"] = json_algo_value;

    // create model data
    json_detect_value["objects"].resize(0);
    for (auto &obj : detected_objects)
    {
        ev::Value tmp_value;
        tmp_value["x"] = int(obj.x1);
        tmp_value["y"] = int(obj.y1);
        tmp_value["width"] = int(obj.x2 - obj.x1);
        tmp_value["height"] = int(obj.y2 - obj.y1);
        tmp_value["name"] = (obj.label > m_config.target_rect_text_map[m_config.language].size() - 1 ? "obj" : m_config.target_rect_text_map[m_config.language][obj.label]);
        tmp_value["confidence"] = obj.score;
        json_detect_value["objects"].append(tmp_value);
        json_algo_value["target_info"].append(tmp_value);
    }
    json_root["model_data"] = json_detect_value;
    json_root["algorithm_data"] = json_algo_value;

    ev::StreamWriterBuilder writer_builder;
    writer_builder.settings_["precision"] = 2;
    writer_builder.settings_["emitUTF8"] = true;
    std::unique_ptr<ev::StreamWriter> json_writer(writer_builder.newStreamWriter());
    std::ostringstream os;
    json_writer->write(json_root, &os);
    m_str_out_json = os.str();
    // 注意：JiEvent.code需要根据需要填充，切勿弄反
    if (is_need_alert)
    {
        event.code = JISDK_CODE_ALARM;
    }
    else
    {
        event.code = JISDK_CODE_NORMAL;
    }
    event.json = m_str_out_json.c_str();
    REC_TIME(t1);
    SDKLOG(INFO) << "output processing time(ms):" << RUN_TIME(t1-t0);

    return JISDK_RET_SUCCEED;
}
