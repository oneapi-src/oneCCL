/*
 Copyright 2016-2020 Intel Corporation
 
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
 
     http://www.apache.org/licenses/LICENSE-2.0
 
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
*/
namespace native {
template struct memory<int8_t, ccl_device, ccl_context>;
template struct memory<uint8_t, ccl_device, ccl_context>;
template struct memory<int16_t, ccl_device, ccl_context>;
template struct memory<uint16_t, ccl_device, ccl_context>;
template struct memory<int32_t, ccl_device, ccl_context>;
template struct memory<uint32_t, ccl_device, ccl_context>;
template struct memory<int64_t, ccl_device, ccl_context>;
template struct memory<uint64_t, ccl_device, ccl_context>;
template struct memory<float, ccl_device, ccl_context>;
template struct memory<double, ccl_device, ccl_context>;
} // namespace native
