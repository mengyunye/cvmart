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


#ifndef COMMON_DET_INFER_H
#define COMMON_DET_INFER_H
#include <memory>
#include <map>
#include <mutex>
#include "opencv2/core.hpp"

#include "evdeploy/deploy.h"

#include "evdeploy/core/ev_common_data.h"
#include "evdeploy/cv/common/common.h"
using namespace ev;

class SampleDetector
{
public:
    SampleDetector();
    ~SampleDetector();
    bool Init(float thresh, const std::string &uuid);
    bool ProcessImage(const cv::Mat &img, std::vector<ev::vision::BoxInfo> &det_results);
    EVStatus YOLOv5Postprocessor(EVMatData *out, std::vector<ev::vision::BoxInfo> &objects, float scale, float thresh, const int img_w, const int img_h);
    bool UnInit();

private:
    vision::YOLOv5Preprocessor m_preprocessor;
    vision::YOLOv5Postprocessor m_postprocessor;

private:
    float m_thresh;
    std::string m_uuid;
    std::mutex m_mutex;
};

#endif
