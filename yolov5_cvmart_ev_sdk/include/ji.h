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

#ifndef EV_JI_H_
#define EV_JI_H_

#include "ji_types.h"
#include "ji_error.h"

#ifdef __cplusplus
extern "C"
{
#endif

/**
 * @brief SDK initialization interface.The authorization function can be implemented in the function
 * @param[in] argc Number of parameters
 * @param[in] argv Parameter array
 * @return JiErrorCode  - operation result
 */
JiErrorCode ji_init(int argc, char **argv);

/**
 * @brief SDK de initialization function
 * @return void         - operation result
 */
void ji_reinit();

/**
 * @brief Create an algorithm predictor instance.
 * 
 * @param[in] pdtype - the predictor type
 * @return void*     - the pionter of algorithm predictor instance
 */
void* ji_create_predictor(JiPredictorType pdtype);

/**
 * @brief Free an algorithm predictor instance.
 * 
 * @param[in] predictor - the predictor instance pointer
 * @return void         - operation result
 */
void ji_destroy_predictor(void *predictor);


/**
 * @brief set callback function
 * 
 * @param[in] predictor - the predictor instance pointer
 * @param[in] callback  - the the callback fucntion
 * @return JiErrorCode         - operation result
 */
JiErrorCode ji_set_callback(void *predictor, JiCallBack callback);

/**
 * @brief Picture analysis asynchronous interface.
 * 
 * @param[in]  predictor  - predictor instance
 * @param[in]  in_frames  - input picture information array
 * @param[in]  in_count   - picture information array size
 * @param[in]  args       - custom algorithm parameters，such as roi
 * @param[in]  user_data   - user data for callbcak
 * @return JiErrorCode    - operation result
 */

JiErrorCode ji_calc_image_asyn(void* predictor, const JiImageInfo* in_frames, const unsigned int in_count, const char* args, void *user_data);

/**
 * @brief Picture analysis synchronous interface.
 * 
 * @param[in]  predictor  - predictor instance
 * @param[in]  in_frames  - input picture information array
 * @param[in]  in_count   - picture information array size
 * @param[in]  args       - custom algorithm parameters，such as roi
 * @param[out] out_frames - output picture information array
 * @param[out] out_count  - output picture information array size
 * @param[out] event      - report algorithm analysis result event
 * @return JiErrorCode    - operation result
 */
JiErrorCode ji_calc_image(void* predictor, const JiImageInfo* in_frames, const unsigned int in_count, const char* args, 
						JiImageInfo **out_frames, unsigned int & out_count, JiEvent &event);

/**
 * @brief Update algorithm configuration.
 *
 * @param[in] predictor - predictor instance
 * @param[in] args      - custom algorithm parameters，such as roi
 * @return JiErrorCode  - operation result
 */
JiErrorCode ji_update_config(void *predictor, const char *args);

/**
 * @brief Get the sdk version.
 *
 * @param[out] version - the current version
 * @return JiErrorCode  - operation result
 */
JiErrorCode ji_get_version(char *version);

/**
 * @brief Create face DB.
 *
 * @param[in] predictor      - predictor instance
 * @param[in] face_db_name     - face DB name
 * @param[in] face_db_id       - face DB id 
 * @param[in] face_db_des      - face DB describe
 * @return JiErrorCode       - operation result
 */
JiErrorCode ji_create_face_db(void *predictor, const char *face_db_name, const int face_db_id, const char *face_db_des);

/**
 * @brief Delete face DB.
 *
 * @param[in] predictor      - predictor instance
 * @param[in] face_db_id       - face DB id 
 * @return JiErrorCode       - operation result
 */
JiErrorCode ji_delete_face_db(void *predictor, const int face_db_id);

/**
 * @brief Get face DB info.
 *
 * @param[in] predictor      - predictor instance
 * @param[in] face_db_id       - face DB id 
 * @param[out] face_db_des     - face info
 * @return JiErrorCode       - operation result
 */
JiErrorCode ji_get_face_db_info(void *predictor, const int face_db_id, char *info);


/**
 * @brief Add face to face DB .
 *
 * @param[in] predictor       - predictor instance
 * @param[in] face_db_id        - face DB id 
 * @param[in] face_name        - face name  
 * @param[in] face_id          - face id 
 * @param[in] data            - face data 
 * @param[in] data_type        - face data type 1 jpg data, 2 image path
 * @param[out] image_path      - output image file full path
 * @return JiErrorCode        - operation result
 */
JiErrorCode ji_face_add(void *predictor, const int face_db_id, const char *face_name, const int face_id, const char *data, const int data_type, char *image_path);

/**
 * @brief Update face to face DB .
 *
 * @param[in] predictor       - predictor instance
 * @param[in] face_db_id        - face DB id 
 * @param[in] face_name        - face name  
 * @param[in] face_id          - face id 
 * @param[in] data            - face data 
 * @param[in] data_type        - face data type 1 jpg data, 2 image path
 * @param[out] image_path      - output image file full path
 * @return JiErrorCode        - operation result
 */
JiErrorCode ji_face_update(void *predictor, const int face_db_id, const char *face_name, const int face_id, const char *data, const int data_type, char *image_path);

/**
 * @brief Delete face in face DB .
 *
 * @param[in] predictor       - predictor instance
 * @param[in] face_db_id        - face DB id 
 * @param[in] face_name        - face name  
 * @param[in] face_id          - face id 
 * @return JiErrorCode        - operation result
 */
JiErrorCode ji_face_delete(void *predictor, const int face_db_id, const int face_id);


#ifdef __cplusplus
}
#endif

#endif // EV_JI_H_